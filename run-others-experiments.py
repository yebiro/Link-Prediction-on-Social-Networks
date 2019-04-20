import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import link_prediction_scores as lp
import pickle, json
import os
import tensorflow as tf


NUM_REPEATS = 1        # 迭代次数
RANDOM_SEED = 0        # 随机种子
#FRAC_EDGES_HIDDEN = [0.15,0.3,0.45]   # 隐藏比例
FRAC_EDGES_HIDDEN = [0.15]
TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'

# 网络数据集
NetWorks=['twitter', 'gplus', 'hamster', 'advogato']
for network in NetWorks:

    network_dir = './data/{}/{}-adj.pkl'.format(network, network)
    network_adj = None
    with open(network_dir, 'rb') as f:
        network_adj = pickle.load(f)

    ### ---------- 运行链路预测实验 ---------- ###
    for i in range(NUM_REPEATS):
        network_results = {}  # 嵌套字典保存实验结果

        # 根据实验命名实验结果文件
        past_results = os.listdir('./results/')
        experiment_num = 0
        experiment_file_name = '{}-experiment-{}-results.json'.format(network, experiment_num)
        while (experiment_file_name in past_results):
            experiment_num += 1
            experiment_file_name = '{}-experiment-{}-results.json'.format(network, experiment_num)

        network_results_dir = './results/' + experiment_file_name

        # 遍历不同隐藏比例的数据集（训练集、验证集、测试集）
        for frac_hidden in FRAC_EDGES_HIDDEN:
            val_frac = 0.05
            test_frac = frac_hidden - val_frac

            # 读取划分好的数据集
            experiment_name = '{}-{}-hidden'.format(network, frac_hidden)
            print("Current experiment: ", experiment_name)
            train_test_split_file = TRAIN_TEST_SPLITS_FOLDER + experiment_name + '.pkl'

            # 在当前网络上运行所有链接预测方法，返回结果
            network_results[experiment_name] = lp.calculate_all_scores(network_adj, features_matrix=None,
                                                                        directed=False, \
                                                                        test_frac=test_frac, val_frac=val_frac, \
                                                                        random_state=RANDOM_SEED, verbose=2,
                                                                        train_test_split_file=train_test_split_file,
                                                                        tf_dtype=tf.float16)

            # 每次遍历保存实验结果
            # json文件
            with open(network_results_dir, 'w') as fp:
                json.dump(network_results, fp, indent=4)

        # 保存最终实验结果
        with open(network_results_dir, 'w') as fp:
            json.dump(network_results, fp, indent=4)