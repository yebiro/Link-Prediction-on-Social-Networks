import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from math import log



##################################
#########   读取边列表   #########
##################################
# 社交网络数据集
NetWorks=['twitter', 'gplus', 'hamster', 'advogato']

for network in NetWorks:
    # 加载边数据
    print('')
    print('Reading {} edgelist'.format(network))
    network_edges_dir = './data/{}/{}.txt'.format(network,network)

    # 无向图
    if network in ['hamster']:
        with open(network_edges_dir, 'rb')as edges_f:
            network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                         data=(('weight', float),))
        adj = nx.adjacency_matrix(network_g)

        # 保存邻接矩阵
        with open('./data/{}/{}-adj.pkl'.format(network,network), 'wb') as f:
            pickle.dump(adj, f)

    # 有向图
    else:
        with open(network_edges_dir, 'rb')as edges_f:
            network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                         data=(('weight', float),))
        print('Num. weakly connected components: ', nx.number_weakly_connected_components(network_g))

        adj = nx.adjacency_matrix(network_g)

        # 保存邻接矩阵
        with open('./data/{}/{}-adj.pkl'.format(network, network), 'wb') as f:
            pickle.dump(adj, f)








