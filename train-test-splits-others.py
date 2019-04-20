import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gae.preprocessing import mask_test_edges_directed, mask_test_edges

RANDOM_SEED = 0
# 数据集
NetWorks=['twitter', 'gplus', 'hamster', 'advogato']
for network in NetWorks:
    network_adj = pickle.load(open('./data/{}/{}-adj.pkl'.format(network, network), 'rb'))

    FRAC_EDGES_HIDDEN = [0.15, 0.3, 0.45]
    TRAIN_TEST_SPLIT_DIR = './train-test-splits/'

    # 遍历隐藏比例来划分数据集（训练集、验证集、测试集）
    for frac_hidden in FRAC_EDGES_HIDDEN:
        val_frac = 0.05
        test_frac = frac_hidden - val_frac

        # 设置随机种子
        np.random.seed(RANDOM_SEED)

        # 生成划分集
        # (adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false)
        # 区分有向图和无向图
        if network in ['hamster']:
            train_test_split = mask_test_edges(network_adj,
                                               test_frac=test_frac, val_frac=val_frac,
                                               verbose=True)
        else:
            train_test_split = mask_test_edges_directed(network_adj,
                                                        test_frac=test_frac, val_frac=val_frac,
                                                        verbose=True, prevent_disconnect=False,
                                                        false_edge_sampling='random')

        file_name = TRAIN_TEST_SPLIT_DIR + '{}-{}-hidden.pkl'.format(network, frac_hidden)
        with open(file_name, 'wb') as f:
            pickle.dump(train_test_split, f, protocol=2)
