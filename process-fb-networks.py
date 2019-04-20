import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle

# facebook 自我中心网络
FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

# 将所有自我中心网络保存为pickle文件作为(adj, features) 元组
for ego_user in FB_EGO_USERS:
    edges_dir = './data/facebook/' + str(ego_user) + '.edges'
    feats_dir = './data/facebook/' + str(ego_user) + '.allfeat'
    
    # 读取边
    with open(edges_dir, 'rb')as f:
        g = nx.read_edgelist(f, nodetype=int, encoding='latin1')

    # 添加自我中心用户（直接连接到所有其他节点）
    g.add_node(ego_user)
    for node in g.nodes():
        if node != ego_user:
            g.add_edge(ego_user, node)

    # 用dataframe保存节点特征
    df = pd.read_table(feats_dir, sep=' ', header=None, index_col=0)

    # 把dataframe中特征添加到 networkx 节点
    for node_index, features_series in df.iterrows():
        # 如果特征对应的节点不在边列表则把节点添加到列表
        if not g.has_node(node_index):
            g.add_node(node_index)
            g.add_edge(node_index, ego_user)

        g.node[node_index]['features'] = features_series.as_matrix()

    assert nx.is_connected(g)
    
    # 以稀疏格式获取邻接矩阵（按g.nodes()排序）
    adj = nx.adjacency_matrix(g) 

    # 获取特征矩阵 (按g.nodes()排序)
    features = np.zeros((df.shape[0], df.shape[1])) # num nodes, num features
    for i, node in enumerate(g.nodes()):
        features[i,:] = g.node[node]['features']

    # 保存邻接矩阵（adj）和特征矩阵（features）到pickle文件
    network_tuple = (adj, features)
    with open("data/fb-processed/{0}-adj-feat.pkl".format(ego_user), "wb") as f:
        pickle.dump(network_tuple, f)