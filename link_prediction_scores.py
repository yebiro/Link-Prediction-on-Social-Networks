
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.manifold import spectral_embedding
import node2vec
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import time
import os
import tensorflow as tf
from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_edges_directed
import pickle
from copy import deepcopy

#sigmod激活函数
def sigmoid(x):
    if x>=0:
        return 1 / (1 + np.exp(-x))
    else:
        return 1 / (1 + np.exp(x))

#绘制训练损失和准确度以及验证AUC值和AP值曲线
def draw_gae_training(dataset, epochs, train_loss, train_acc, val_roc, val_ap):
    # plot the training loss and accuracy

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0, epochs), train_loss, label="train_loss")
    ax2.plot(np.arange(0, epochs), train_acc, label="train_accuracy", color='r')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('train accuracy')
    plt.legend(['train_loss', 'train_accuracy'], loc="center right")
    plt.savefig("results/tables/{}_loss_accuracy.png".format(dataset))
    plt.show()



    plt.plot(np.arange(0, epochs), val_roc, label="val_auc")
    plt.xlabel("Epoch")
    plt.ylabel("Area under Curve")
    plt.legend(loc="center right")
    plt.savefig("results/tables/{}_val_roc.png".format(dataset))
    plt.show()


    plt.plot(np.arange(0, epochs), val_ap, label="val_ap")
    # plt.title("Training Loss and Accuracy on sar classifier")
    plt.xlabel("Epoch")
    plt.ylabel("Average Accuracy")
    plt.legend(loc="center right")
    plt.savefig("results/tables/{}_val_ap.png".format(dataset))
    plt.show()

# 输入: positive test/val edges, negative test/val edges, edge score matrix
# 输出: ROC AUC score, ROC Curve (FPR, TPR, Thresholds), AP score
def get_roc_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=False):

    # 边的情况
    if len(edges_pos) == 0 or len(edges_neg) == 0:
        return (None, None, None)

    # 保存正例边预测，实际值为1
    preds_pos = []
    pos = []
    for edge in edges_pos:
        if apply_sigmoid == True:
            preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_pos.append(score_matrix[edge[0], edge[1]])
        pos.append(1) # 1-正例边
        
    # 保存负例边预测，实际值为0
    preds_neg = []
    neg = []
    for edge in edges_neg:
        if apply_sigmoid == True:
            preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))
        else:
            preds_neg.append(score_matrix[edge[0], edge[1]])
        neg.append(0) # 0-负例边
        
    # 计算得分
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    # roc_curve_tuple = roc_curve(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    # 返回 roc_score, ap_score
    return roc_score, ap_score

# 返回(node1, node2)元组列表，用于networkx链路预测评估
def get_ebunch(train_test_split):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split
 
    test_edges_list = test_edges.tolist() # 转换为嵌套列表
    test_edges_list = [tuple(node_pair) for node_pair in test_edges_list] # 把节点对转换为元组
    test_edges_false_list = test_edges_false.tolist()
    test_edges_false_list = [tuple(node_pair) for node_pair in test_edges_false_list]
    return (test_edges_list + test_edges_false_list)

# 输入: NetworkX 训练图, train_test_split (通过mask_test_edges划分)
# 输出: AA方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def adamic_adar_scores(g_train, train_test_split):
    if g_train.is_directed(): # 只用于无向图，如果是有向图，将其转为无向图
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # 加载输入划分集

    start_time = time.time()
    
    aa_scores = {}

    # 计算得分
    aa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.adamic_adar_index(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = 节点索引, p = AA 指数
        aa_matrix[u][v] = p
        aa_matrix[v][u] = p # 确保它是对称的
    aa_matrix = aa_matrix / aa_matrix.max() # 归一化矩阵

    runtime = time.time() - start_time
    aa_roc, aa_ap = get_roc_score(test_edges, test_edges_false, aa_matrix)

    aa_scores['test_roc'] = aa_roc
    # aa_scores['test_roc_curve'] = aa_roc_curve
    aa_scores['test_ap'] = aa_ap
    aa_scores['runtime'] = runtime
    return aa_scores


# 输入: NetworkX 训练图, train_test_split (通过mask_test_edges划分)
# 输出: JC方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def jaccard_coefficient_scores(g_train, train_test_split):
    if g_train.is_directed(): # 只用于无向图，如果是有向图，将其转为无向图
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # 加载输入划分集

    start_time = time.time()
    jc_scores = {}

    # 计算得分
    jc_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.jaccard_coefficient(g_train, ebunch=get_ebunch(train_test_split)): # (u, v) = 节点索引, p = JC 指数
        jc_matrix[u][v] = p
        jc_matrix[v][u] = p # 确保它是对称的
    jc_matrix = jc_matrix / jc_matrix.max() # 归一化矩阵

    runtime = time.time() - start_time
    jc_roc, jc_ap = get_roc_score(test_edges, test_edges_false, jc_matrix)

    jc_scores['test_roc'] = jc_roc
    # jc_scores['test_roc_curve'] = jc_roc_curve
    jc_scores['test_ap'] = jc_ap
    jc_scores['runtime'] = runtime
    return jc_scores


# 输入: NetworkX 训练图, train_test_split (通过mask_test_edges划分)
# 输出: PA方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def preferential_attachment_scores(g_train, train_test_split):
    if g_train.is_directed():  # 只用于无向图，如果是有向图，将其转为无向图
        g_train = g_train.to_undirected()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # 加载输入划分集

    start_time = time.time()
    pa_scores = {}

    # 计算得分
    pa_matrix = np.zeros(adj_train.shape)
    for u, v, p in nx.preferential_attachment(g_train, ebunch=get_ebunch(train_test_split)): #  (u, v) = 节点索引, p = PA 指数
        pa_matrix[u][v] = p
        pa_matrix[v][u] = p # 确保它是对称的
    pa_matrix = pa_matrix / pa_matrix.max() # 归一化矩阵

    runtime = time.time() - start_time
    pa_roc, pa_ap = get_roc_score(test_edges, test_edges_false, pa_matrix)

    pa_scores['test_roc'] = pa_roc
    #pa_scores['test_roc_curve'] = pa_roc_curve
    pa_scores['test_ap'] = pa_ap
    pa_scores['runtime'] = runtime
    return pa_scores


# 输入: train_test_split (通过mask_test_edges划分)
# 输出: PA方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def spectral_clustering_scores(train_test_split, random_state=0):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split  # 加载输入划分集

    start_time = time.time()
    sc_scores = {}

    # 进行谱聚类链接预测
    spectral_emb = spectral_embedding(adj_train, n_components=16, random_state=random_state)
    sc_score_matrix = np.dot(spectral_emb, spectral_emb.T)

    runtime = time.time() - start_time
    sc_test_roc, sc_test_ap = get_roc_score(test_edges, test_edges_false, sc_score_matrix, apply_sigmoid=True)
    sc_val_roc, sc_val_ap = get_roc_score(val_edges, val_edges_false, sc_score_matrix, apply_sigmoid=True)

    # 记录得分
    sc_scores['test_roc'] = sc_test_roc
    # sc_scores['test_roc_curve'] = sc_test_roc_curve
    sc_scores['test_ap'] = sc_test_ap

    sc_scores['val_roc'] = sc_val_roc
    # sc_scores['val_roc_curve'] = sc_val_roc_curve
    sc_scores['val_ap'] = sc_val_ap

    sc_scores['runtime'] = runtime
    return sc_scores

# 输入: NetworkX 训练图， train_test_split (通过mask_test_edges划分)，Node2Vec超参数
# 输出: Node2Vec方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def node2vec_scores(
    g_train, train_test_split,
    P = 1, # 返回概率参数
    Q = 1, # 进出概率参数
    WINDOW_SIZE = 10, # 优化的上下文大小
    NUM_WALKS = 10, # 每次源的游走次数
    WALK_LENGTH = 80, # 每次源的游走序列长度
    DIMENSIONS = 128, # 嵌入维度
    DIRECTED = False, # 有向/无向图
    WORKERS = 8, # 平行游者的数量
    ITER = 1, #  SGD 迭代次数
    edge_score_mode = "edge-emb", # 使用自举边嵌入+逻辑回归,
        # 或者使用简单的点积用于计算边得分
    verbose=1,
    ):
    if g_train.is_directed():
        DIRECTED = True

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    start_time = time.time()

    # 预处理,生成 walks
    if verbose >= 1:
        print('Preprocessing grpah for node2vec...')
    g_n2v = node2vec.Graph(g_train, DIRECTED, P, Q) # 创建 node2vec 图实例
    g_n2v.preprocess_transition_probs()
    if verbose == 2:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=True)
    else:
        walks = g_n2v.simulate_walks(NUM_WALKS, WALK_LENGTH, verbose=False)
    walks = [list(map(str, walk)) for walk in walks]

    # 训练skip-gram模型
    model = Word2Vec(walks, size=DIMENSIONS, window=WINDOW_SIZE, min_count=0, sg=1, workers=WORKERS, iter=ITER)

    # 保存嵌入映射
    emb_mappings = model.wv

    # 创建节点嵌入矩阵(rows = nodes, columns = embedding features)
    emb_list = []
    for node_index in range(0, adj_train.shape[0]):
        node_str = str(node_index)
        node_emb = emb_mappings[node_str]
        emb_list.append(node_emb)
    emb_matrix = np.vstack(emb_list)

    # 生成自举边嵌入 (按照node2vec论文做法)
    # (v1, v2)的边嵌入 = v1，v2节点嵌入的哈马达积
    if edge_score_mode == "edge-emb":
        
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = emb_matrix[node1]
                emb2 = emb_matrix[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # 训练集 边嵌入
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # 创建训练集边标签： 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # 验证集 边嵌入 标签
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # 测试集 边嵌入 标签
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # 创建验证集边标签： 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # 在训练集边嵌入上训练逻辑回归分类器
        edge_classifier = LogisticRegression(random_state=0, solver='liblinear')
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        # 预测边的得分：分为1类（真实边）的概率
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        runtime = time.time() - start_time

        # 计算得分
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            n2v_val_roc = roc_auc_score(val_edge_labels, val_preds)
            # n2v_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            n2v_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        n2v_test_roc = roc_auc_score(test_edge_labels, test_preds)
        # n2v_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        n2v_test_pr_curve = precision_recall_curve(test_edge_labels, test_preds)
        n2v_test_ap = average_precision_score(test_edge_labels, test_preds)


    # 使用节点嵌入的简单点积生成边得分
    elif edge_score_mode == "dot-product":
        score_matrix = np.dot(emb_matrix, emb_matrix.T)
        runtime = time.time() - start_time

        # 验证集得分
        if len(val_edges) > 0:
            n2v_val_roc, n2v_val_ap = get_roc_score(val_edges, val_edges_false, score_matrix, apply_sigmoid=True)
        else:
            n2v_val_roc = None
            n2v_val_roc_curve = None
            n2v_val_ap = None
        
        # 测试集得分
        n2v_test_roc, n2v_test_ap = get_roc_score(test_edges, test_edges_false, score_matrix, apply_sigmoid=True)

    else:
        print("Invalid edge_score_mode! Either use edge-emb or dot-product.")

    # 记录得分
    n2v_scores = {}

    n2v_scores['test_roc'] = n2v_test_roc
    # n2v_scores['test_roc_curve'] = n2v_test_roc_curve
    # n2v_scores['test_pr_curve'] = n2v_test_pr_curve
    n2v_scores['test_ap'] = n2v_test_ap

    n2v_scores['val_roc'] = n2v_val_roc
    # n2v_scores['val_roc_curve'] = n2v_val_roc_curve
    n2v_scores['val_ap'] = n2v_val_ap

    n2v_scores['runtime'] = runtime

    return n2v_scores



# 输入: 原始稀疏邻接矩阵adj_sparse， train_test_split (通过mask_test_edges划分)，特征矩阵，VAGE
# 输出: VGAE方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def gae_scores(
    adj_sparse,
    train_test_split,
    features_matrix=None,
    LEARNING_RATE = 0.01,
    EPOCHS = 250,
    HIDDEN1_DIM = 32,
    HIDDEN2_DIM = 16,
    DROPOUT = 0,
    edge_score_mode="dot-product",
    verbose=1,
    dtype=tf.float32
    ):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split # Unpack train-test split

    if verbose >= 1:
        print('GAE preprocessing...')

    # start_time = time.time()

    # 由于内存限制，使用CPU (隐藏 GPU)训练
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # 特征转换 正常矩阵 --> 稀疏矩阵 --> 元组
    # 特征元组包含： (矩阵坐标列表, 矩阵值列表, 矩阵维度)
    if features_matrix is None:
        x = sp.lil_matrix(np.identity(adj_sparse.shape[0]))
    else:
        x = sp.lil_matrix(features_matrix)
    features_tuple = sparse_to_tuple(x)
    features_shape = features_tuple[2]

    # 获取图属性 (用于输入模型)
    num_nodes = adj_sparse.shape[0] # 邻接矩阵的节点数量
    num_features = features_shape[1] # 特征数量 (特征矩阵的列数)
    features_nonzero = features_tuple[1].shape[0] # 特征矩阵中的非零条目数(或者矩阵值列表长度)

    # 保存原始邻接矩阵 (没有对角线条目) 到后面使用
    adj_orig = deepcopy(adj_sparse)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    # 归一化邻接矩阵
    adj_norm = preprocess_graph(adj_train)

    # 添加对角线
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # 定义占位符
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # How much to weigh positive examples (true edges) in cost print_function
      # Want to weigh less-frequent classes higher, so as to prevent model output bias
      # pos_weight = (num. negative samples / (num. positive samples)
    pos_weight = float(adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) / adj_sparse.sum()

    # normalize (scale) average weighted cost
    norm = adj_sparse.shape[0] * adj_sparse.shape[0] / float((adj_sparse.shape[0] * adj_sparse.shape[0] - adj_sparse.sum()) * 2)

    if verbose >= 1:
        print('Initializing GAE model...')

    # 创建 VAE 模型
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero,
                       HIDDEN1_DIM, HIDDEN2_DIM, dtype=dtype, flatten_output=False)

    opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
                               # labels=placeholders['adj_orig'],
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm,
                               learning_rate=LEARNING_RATE,
                               dtype=tf.float32)

    cost_val = []
    acc_val = []
    val_roc_score = []

    prev_embs = []

    # 初始化 session
    sess = tf.Session()

    if verbose >= 1:
        # 打印所有可训练的变量
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape 是tf.Dimension的一个数组
            shape = variable.get_shape()
            print("Variable shape: ", shape)
            variable_parameters = 1
            for dim in shape:
                print("Current dimension: ", dim)
                variable_parameters *= dim.value
            print("Variable params: ", variable_parameters)
            total_parameters += variable_parameters
            print('')
        print("TOTAL TRAINABLE PARAMS: ", total_parameters)

        print('Initializing TF variables...')

    sess.run(tf.global_variables_initializer())

    if verbose >= 1:
        print('Starting GAE training!')

    start_time = time.time()
    # 训练模型
    train_loss = []
    train_acc = []
    val_roc = []
    val_ap = []
    for epoch in range(EPOCHS):

        t = time.time()
        # 构造 feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
        feed_dict.update({placeholders['dropout']: DROPOUT})
        # 单一权重更新
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # 计算平均损失
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        # 评估预测
        feed_dict.update({placeholders['dropout']: 0})
        gae_emb = sess.run(model.z_mean, feed_dict=feed_dict)

        prev_embs.append(gae_emb)

        gae_score_matrix = np.dot(gae_emb, gae_emb.T)



        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false, gae_score_matrix, apply_sigmoid=True)
        val_roc_score.append(roc_curr)

        # 每次迭代打印结果
        # if verbose == 2:
        #     print(("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
        #       "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
        #       "val_ap=", "{:.5f}".format(ap_curr),
        #       "time=", "{:.5f}".format(time.time() - t)))

        train_loss.append(avg_cost)
        train_acc.append(avg_accuracy)
        val_roc.append(val_roc_score[-1])
        val_ap.append(ap_curr)

    # 画出训练过程损失和准确度以及验证AUC和AP
    #draw_gae_training('hamster', EPOCHS, train_loss, train_acc, val_roc, val_ap)
    runtime = time.time() - start_time
    if verbose == 2:
        print("Optimization Finished!")

    # 打印最终结果
    feed_dict.update({placeholders['dropout']: 0})
    gae_emb = sess.run(model.z_mean, feed_dict=feed_dict)

    # 点积边得分
    if edge_score_mode == "dot-product":
        gae_score_matrix = np.dot(gae_emb, gae_emb.T)

        # runtime = time.time() - start_time

        # 计算最终得分
        gae_val_roc, gae_val_ap = get_roc_score(val_edges, val_edges_false, gae_score_matrix)
        gae_test_roc, gae_test_ap = get_roc_score(test_edges, test_edges_false, gae_score_matrix)
    
    # 采取自举边嵌入 (通过哈达玛积)
    elif edge_score_mode == "edge-emb":
        def get_edge_embeddings(edge_list):
            embs = []
            for edge in edge_list:
                node1 = edge[0]
                node2 = edge[1]
                emb1 = gae_emb[node1]
                emb2 = gae_emb[node2]
                edge_emb = np.multiply(emb1, emb2)
                embs.append(edge_emb)
            embs = np.array(embs)
            return embs

        # 训练集 边嵌入
        pos_train_edge_embs = get_edge_embeddings(train_edges)
        neg_train_edge_embs = get_edge_embeddings(train_edges_false)
        train_edge_embs = np.concatenate([pos_train_edge_embs, neg_train_edge_embs])

        # 创建训练集 边标签： 1 = real edge, 0 = false edge
        train_edge_labels = np.concatenate([np.ones(len(train_edges)), np.zeros(len(train_edges_false))])

        # 验证集 边嵌入，标签
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            pos_val_edge_embs = get_edge_embeddings(val_edges)
            neg_val_edge_embs = get_edge_embeddings(val_edges_false)
            val_edge_embs = np.concatenate([pos_val_edge_embs, neg_val_edge_embs])
            val_edge_labels = np.concatenate([np.ones(len(val_edges)), np.zeros(len(val_edges_false))])

        # 测试集 边嵌入，标签
        pos_test_edge_embs = get_edge_embeddings(test_edges)
        neg_test_edge_embs = get_edge_embeddings(test_edges_false)
        test_edge_embs = np.concatenate([pos_test_edge_embs, neg_test_edge_embs])

        # 创建验证集 边标签： 1 = real edge, 0 = false edge
        test_edge_labels = np.concatenate([np.ones(len(test_edges)), np.zeros(len(test_edges_false))])

        # 在训练集边嵌入上训练逻辑回归分类器
        edge_classifier = LogisticRegression(random_state=0, solver='liblinear')
        edge_classifier.fit(train_edge_embs, train_edge_labels)

        #预测边得分: 分为1类（真实边）的概率
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            val_preds = edge_classifier.predict_proba(val_edge_embs)[:, 1]
        test_preds = edge_classifier.predict_proba(test_edge_embs)[:, 1]

        #runtime = time.time() - start_time

        # 计算得分
        if len(val_edges) > 0 and len(val_edges_false) > 0:
            gae_val_roc = roc_auc_score(val_edge_labels, val_preds)
            gae_val_roc_curve = roc_curve(val_edge_labels, val_preds)
            gae_val_ap = average_precision_score(val_edge_labels, val_preds)
        else:
            gae_val_roc = None
            gae_val_roc_curve = None
            gae_val_ap = None
        
        gae_test_roc = roc_auc_score(test_edge_labels, test_preds)
        gae_test_roc_curve = roc_curve(test_edge_labels, test_preds)
        gae_test_pr_curve = precision_recall_curve(test_edge_labels, test_preds)
        gae_test_ap = average_precision_score(test_edge_labels, test_preds)

    # 记录得分
    gae_scores = {}

    gae_scores['test_roc'] = gae_test_roc
    gae_scores['test_ap'] = gae_test_ap

    gae_scores['val_roc'] = gae_val_roc
    gae_scores['val_ap'] = gae_val_ap

    if(edge_score_mode=="edge-emb"):
        gae_scores['test_roc_curve'] = gae_test_roc_curve
        gae_scores['val_roc_curve'] = gae_val_roc_curve
        gae_scores['test_pr_curve'] = gae_test_pr_curve

    gae_scores['val_roc_per_epoch'] = val_roc_score
    gae_scores['runtime'] = runtime
    return gae_scores
    


# 输入: adj_sparse（邻接矩阵，以稀疏矩阵形式）, features_matrix（特征矩阵）, test_frac（测试集比例）, val_frac（验证集比例）, verbose（是否显示详细过程）
    # Verbose: 0 - 不打印, 1 - 打印得分, 2 - 打印得分 + GAE 训练过程
# 返回: 每个链路预测方法的结果字典(ROC AUC, ROC Curve, AP, Runtime)
def calculate_all_scores(adj_sparse, features_matrix=None, directed=False, \
        test_frac=.1, val_frac=.05, random_state=0, verbose=1, \
        train_test_split_file=None,
        tf_dtype=tf.float32):
    np.random.seed(random_state) # Guarantee consistent train/test split
    tf.set_random_seed(random_state) # Consistent GAE training

    # 链路预测得分字典
    lp_scores = {}

    ### ---------- 预处理 ---------- ###
    train_test_split = None
    try: # 如果找到存在的划分好的数据集，则使用找到的文件
        with open(train_test_split_file, 'rb') as f:
            train_test_split = pickle.load(f)
            print('Found existing train-test split!')
    except: # 否则, 生成数据划分集
        print('Generating train-test split...')
        if directed == False:
            train_test_split = mask_test_edges(adj_sparse, test_frac=test_frac, val_frac=val_frac)
        else:
            train_test_split = mask_test_edges_directed(adj_sparse, test_frac=test_frac, val_frac=val_frac)
    
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = train_test_split  # 打开元组

    # g_train: 完整的图对象（没有隐藏边）
    if directed == True:
        g_train = nx.DiGraph(adj_train)
    else:
        g_train = nx.Graph(adj_train)

    # 检查训练集测试集划分
    if verbose >= 1:
        print("Total nodes:", adj_sparse.shape[0])
        print("Total edges:", int(adj_sparse.nnz/2)) # adj is symmetric, so nnz (num non-zero) = 2*num_edges
        print("Training edges (positive):", len(train_edges))
        print("Training edges (negative):", len(train_edges_false))
        print("Validation edges (positive):", len(val_edges))
        print("Validation edges (negative):", len(val_edges_false))
        print("Test edges (positive):", len(test_edges))
        print("Test edges (negative):", len(test_edges_false))
        print('')
        print("------------------------------------------------------")


    # ---------- 链路预测基线方法---------- ###
    # # Adamic-Adar
    aa_scores = adamic_adar_scores(g_train, train_test_split)
    lp_scores['aa'] = aa_scores
    if verbose >= 1:
        print('')
        print('Adamic-Adar Test ROC score: ', str(aa_scores['test_roc']))
        print('Adamic-Adar Test AP score: ', str(aa_scores['test_ap']))

    # Jaccard Coefficient
    jc_scores = jaccard_coefficient_scores(g_train, train_test_split)
    lp_scores['jc'] = jc_scores
    if verbose >= 1:
        print('')
        print('Jaccard Coefficient Test ROC score: ', str(jc_scores['test_roc']))
        print('Jaccard Coefficient Test AP score: ', str(jc_scores['test_ap']))

    # Preferential Attachment
    pa_scores = preferential_attachment_scores(g_train, train_test_split)
    lp_scores['pa'] = pa_scores
    if verbose >= 1:
        print('')
        print('Preferential Attachment Test ROC score: ', str(pa_scores['test_roc']))
        print('Preferential Attachment Test AP score: ', str(pa_scores['test_ap']))


    ### ---------- SPECTRAL CLUSTERING ---------- ###
    sc_scores = spectral_clustering_scores(train_test_split)
    lp_scores['sc'] = sc_scores
    if verbose >= 1:
        print('')
        print('Spectral Clustering Validation ROC score: ', str(sc_scores['val_roc']))
        print('Spectral Clustering Validation AP score: ', str(sc_scores['val_ap']))
        print('Spectral Clustering Test ROC score: ', str(sc_scores['test_roc']))
        print('Spectral Clustering Test AP score: ', str(sc_scores['test_ap']))
        print('')

    ## ---------- NODE2VEC ---------- ###
    # node2vec 参数设置
    # 当 p = q = 1, Node2Vec等同于DeepWalk
    P = 1 # 返回概率参数
    Q = 1 # 进出概率参数
    WINDOW_SIZE = 10 # 优化的上下文大小
    NUM_WALKS = 10 # 每次源的游走次数
    WALK_LENGTH = 80 # 每次源的游走序列长度
    DIMENSIONS = 128 # 嵌入维度
    DIRECTED = False # 有向/无向图
    WORKERS = 8 # 平行游者的数量
    ITER = 1 # SGD 迭代次数

    # 使用自举边嵌入+逻辑回归
    n2v_edge_emb_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "edge-emb",
        verbose)
    lp_scores['n2v_edge_emb'] = n2v_edge_emb_scores

    if verbose >= 1:
        print('')
        print('node2vec (Edge Embeddings) Validation ROC score: ', str(n2v_edge_emb_scores['val_roc']))
        print('node2vec (Edge Embeddings) Validation AP score: ', str(n2v_edge_emb_scores['val_ap']))
        print('node2vec (Edge Embeddings) Test ROC score: ', str(n2v_edge_emb_scores['test_roc']))
        print('node2vec (Edge Embeddings) Test AP score: ', str(n2v_edge_emb_scores['test_ap']))
        print('')

    # 使用点积计算边得分
    n2v_dot_prod_scores = node2vec_scores(g_train, train_test_split,
        P, Q, WINDOW_SIZE, NUM_WALKS, WALK_LENGTH, DIMENSIONS, DIRECTED, WORKERS, ITER,
        "dot-product",
        verbose)
    lp_scores['n2v_dot_prod'] = n2v_dot_prod_scores

    if verbose >= 1:
        print('')
        print('node2vec (Dot Product) Validation ROC score: ', str(n2v_dot_prod_scores['val_roc']))
        print('node2vec (Dot Product) Validation AP score: ', str(n2v_dot_prod_scores['val_ap']))
        print('node2vec (Dot Product) Test ROC score: ', str(n2v_dot_prod_scores['test_roc']))
        print('node2vec (Dot Product) Test AP score: ', str(n2v_dot_prod_scores['test_ap']))
        print('')


    ### ---------- (VARIATIONAL) GRAPH AUTOENCODER ---------- ###
    # # GAE 参数设置
    LEARNING_RATE = 0.01  # Default: 0.01
    EPOCHS = 250
    HIDDEN1_DIM = 32
    HIDDEN2_DIM = 16
    DROPOUT = 0

    #  使用点积
    tf.set_random_seed(random_state)  # Consistent GAE training
    gae_results = gae_scores(adj_sparse, train_test_split, features_matrix,
                             LEARNING_RATE, EPOCHS, HIDDEN1_DIM, HIDDEN2_DIM, DROPOUT,
                             "dot-product",
                             verbose,
                             dtype=tf.float32)
    lp_scores['gae'] = gae_results

    if verbose >= 1:
        print('')
        print('GAE (Dot Product) Validation ROC score: ', str(gae_results['val_roc']))
        print('GAE (Dot Product) Validation AP score: ', str(gae_results['val_ap']))
        print('GAE (Dot Product) Test ROC score: ', str(gae_results['test_roc']))
        print('GAE (Dot Product) Test AP score: ', str(gae_results['test_ap']))
        print("------------------------------------------------------")
        print("------------------------------------------------------")
        print('')


    # 使用边嵌入
    tf.set_random_seed(random_state) # Consistent GAE training
    gae_edge_emb_results = gae_scores(adj_sparse, train_test_split, features_matrix,
        LEARNING_RATE, EPOCHS, HIDDEN1_DIM, HIDDEN2_DIM, DROPOUT,
        "edge-emb",
        verbose)
    lp_scores['gae_edge_emb'] = gae_edge_emb_results

    if verbose >= 1:
        print('')
        print('GAE (Edge Embeddings) Validation ROC score: ', str(gae_edge_emb_results['val_roc']))
        print('GAE (Edge Embeddings) Validation AP score: ', str(gae_edge_emb_results['val_ap']))
        #print('GAE (Edge Embeddings) Validation ROC_CURVE score: ', str(gae_edge_emb_results['val_roc_curve']))
        print('GAE (Edge Embeddings) Test ROC score: ', str(gae_edge_emb_results['test_roc']))
        print('GAE (Edge Embeddings) Test AP score: ', str(gae_edge_emb_results['test_ap']))
        #print('GAE (Edge Embeddings) Test ROC_CURVE score: ', str(gae_edge_emb_results['test_roc_curve']))


    ### ---------- 返回结果 ---------- ###
    return lp_scores
