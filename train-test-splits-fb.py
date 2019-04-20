import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gae.preprocessing import mask_test_edges

RANDOM_SEED = 0

### ---------- 读取Facebook网络数据 ---------- ###

FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
fb_graphs = {}  # 保存所有的FB自我中心网络

# 读取每个FB自我中心网络
for user in FB_EGO_USERS:
    network_dir = './data/fb-processed/{0}-adj-feat.pkl'.format(user)
    with open(network_dir, 'rb') as f:
        adj, features = pickle.load(f)

    # 保存在字典目录
    fb_graphs[user] = (adj, features)
    
# 读取由10个自我中心网络结合的FB网络
combined_dir = './data/fb-processed/combined-adj-sparsefeat.pkl'
with open(combined_dir, 'rb') as f:
    adj, features = pickle.load(f)
    fb_graphs['combined'] = (adj, features)


### ---------- 生成 Train-Test Splits ---------- ###
FRAC_EDGES_HIDDEN = [0.15, 0.3, 0.45]
TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'


# 遍历隐藏比例来划分数据集（训练集、验证集、测试集）
for frac_hidden in FRAC_EDGES_HIDDEN:
    val_frac = 0.05
    test_frac = frac_hidden - val_frac
    
    # 遍历每个网络进行划分
    for g_name, graph_tuple in fb_graphs.items():
        adj = graph_tuple[0]
        feat = graph_tuple[1]
        
        current_graph = 'fb-{}-{}-hidden'.format(g_name, frac_hidden)
        
        # 输入当前网络
        print("Current graph: ", current_graph)

        np.random.seed(RANDOM_SEED)
        
        # 调用函数划分
        train_test_split = mask_test_edges(adj, test_frac=test_frac, val_frac=val_frac,
            verbose=True)

        file_name = TRAIN_TEST_SPLITS_FOLDER + current_graph + '.pkl'

        # 保存划分数据
        with open(file_name, 'wb') as f:
            pickle.dump(train_test_split, f, protocol=2)


