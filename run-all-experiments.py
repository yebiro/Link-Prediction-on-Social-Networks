import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import link_prediction_scores as lp
import pickle
import os
import json

NUM_REPEATS = 1       # 迭代次数
RANDOM_SEED = 0       # 随机种子
FRAC_EDGES_HIDDEN = [0.15]  # 隐藏比例

# facebook上链路预测实验
def facebook_networks():
    ###---------- 读取Facebook网络数据----------###
    FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
    fb_graphs = {}   # 保存所有的FB自我中心网络

    # 读取每个FB自我中心网络
    for user in FB_EGO_USERS:
        network_dir = './data/fb-processed/{0}-adj-feat.pkl'.format(user)
        with open(network_dir, 'rb') as f:
            adj, features = pickle.load(f)

        # 保存在字典目录
        fb_graphs[user] = (adj, features)

    # 读取FB-combined网络
    combined_dir = './data/fb-processed/combined-adj-sparsefeat.pkl'
    with open(combined_dir, 'rb') as f:
        adj, features = pickle.load(f)
        fb_graphs['combined'] = (adj, features)

    ### ---------- 运行链路预测实验 ---------- ###
    for i in range(NUM_REPEATS):

        fb_results = {}

        # 根据实验命名实验结果文件
        past_results = os.listdir('./results/')
        txt_past_results = os.listdir('./results/txt/')
        experiment_num = 0
        experiment_file_name = 'fb-experiment-{}-results.pkl'.format(experiment_num)
        while (experiment_file_name in past_results):
            experiment_num += 1
            experiment_file_name = 'fb-experiment-{}-results.pkl'.format(experiment_num)

        FB_RESULTS_DIR = './results/' + experiment_file_name

        # 保存为txt文件
        txt_experiment_num = 0
        txt_experiment_file_name = 'txt-fb-experiment-{}-results.json'.format(txt_experiment_num)
        while (txt_experiment_file_name in txt_past_results):
            txt_experiment_num += 1
            txt_experiment_file_name = 'txt-fb-experiment-{}-results.json'.format(txt_experiment_num)

        TXT_FB_RESULTS_DIR = './results/txt/' + txt_experiment_file_name

        TRAIN_TEST_SPLITS_FOLDER = './train-test-splits/'

        # 遍历不同隐藏比例的数据集（训练集、验证集、测试集）
        for frac_hidden in FRAC_EDGES_HIDDEN:
            val_frac = 0.05
            test_frac = frac_hidden - val_frac

            # 遍历每个网络集
            for g_name, graph_tuple in fb_graphs.items():
                adj = graph_tuple[0]
                feat = graph_tuple[1]

                experiment_name = 'fb-{}-{}-hidden'.format(g_name, frac_hidden)
                print("Current experiment: ", experiment_name)


                train_test_split_file = TRAIN_TEST_SPLITS_FOLDER + experiment_name + '.pkl'

                # 在当前网络上运行所有链接预测方法，返回结果
                fb_results[experiment_name] = lp.calculate_all_scores(adj, feat, \
                                                                      test_frac=test_frac, val_frac=val_frac, \
                                                                      random_state=RANDOM_SEED, verbose=2,
                                                                      train_test_split_file=train_test_split_file)

                # 每次遍历保存实验结果
                # pickle文件
                with open(FB_RESULTS_DIR, 'wb') as f:
                    pickle.dump(fb_results, f, protocol=2)

                # json文件
                with open(TXT_FB_RESULTS_DIR, 'w') as f:
                    json.dump(fb_results, f, indent=4)

        # 保存最终实验结果
        # pickle文件
        with open(FB_RESULTS_DIR, 'wb') as f:
            pickle.dump(fb_results, f, protocol=2)

        # json文件
        with open(TXT_FB_RESULTS_DIR, 'w') as f:
            json.dump(fb_results, f, indent=4)

# 随机网络上链路预测实验
def random_networks():
    ### ---------- 生成随机网络图 ---------- ###
    # 保存生成的随机网络
    nx_graphs = {}


    #NUM=[10, 100, 1000, 10000]
    NUM = [2000]
    for N_LARGE in NUM:
        # N_LARGE = 1000
        # nx_graphs['er-large'] = nx.erdos_renyi_graph(n=N_LARGE, p=.03, seed=RANDOM_SEED) # Erdos-Renyi
        # nx_graphs['ws-large'] = nx.watts_strogatz_graph(n=N_LARGE, k=11, p=.1, seed=RANDOM_SEED)  # Watts-Strogatz
        nx_graphs['ba-large'] = nx.barabasi_albert_graph(n=N_LARGE, m=6, seed=RANDOM_SEED)  # Barabasi-Albert
        # nx_graphs['pc-large'] = nx.powerlaw_cluster_graph(n=N_LARGE, m=6, p=.02, seed=RANDOM_SEED) # Powerlaw Cluster
        # nx_graphs['sbm-large'] = nx.random_partition_graph(sizes=[N_LARGE//10]*10, p_in=.05, p_out=.005, seed=RANDOM_SEED) # Stochastic Block Model

        # 移除孤立点
        for g_name, nx_g in nx_graphs.items():
            isolates = nx.isolates(nx_g)
            if len(list(isolates)) > 0:
                for isolate_node in isolates:
                    nx_graphs[g_name].remove_node(isolate_node)

        ### ---------- 运行链路预测实验 ---------- ###
        for i in range(NUM_REPEATS):
            ## ---------- NETWORKX ---------- ###
            nx_results = {}

            # 根据实验命名实验结果文件
            past_results = os.listdir('./results/')
            txt_past_results = os.listdir('./results/txt/')
            experiment_num = 0
            experiment_file_name = 'nx-experiment-{}-results.pkl'.format(experiment_num)
            while (experiment_file_name in past_results):
                experiment_num += 1
                experiment_file_name = 'nx-experiment-{}-results.pkl'.format(experiment_num)

            NX_RESULTS_DIR = './results/' + experiment_file_name

            # 保存为txt文件
            txt_experiment_num = 0
            txt_experiment_file_name = 'txt-nx-experiment-{}-results.json'.format(txt_experiment_num)
            while (txt_experiment_file_name in txt_past_results):
                txt_experiment_num += 1
                txt_experiment_file_name = 'txt-nx-experiment-{}-results.json'.format(txt_experiment_num)

            TXT_NX_RESULTS_DIR = './results/txt/' + txt_experiment_file_name

            # 遍历不同隐藏比例的数据集（训练集、验证集、测试集）
            for frac_hidden in FRAC_EDGES_HIDDEN:
                val_frac = 0.05
                test_frac = frac_hidden - val_frac

                # 遍历每个随机网络
                for g_name, nx_g in nx_graphs.items():
                    adj = nx.adjacency_matrix(nx_g)

                    experiment_name = 'nx-{}-{}-hidden'.format(g_name, frac_hidden)
                    print("Current experiment: ", experiment_name)

                    # 在当前网络上运行所有链接预测方法，返回结果
                    nx_results[experiment_name] = lp.calculate_all_scores(adj, \
                                                                          test_frac=test_frac, val_frac=val_frac, \
                                                                          random_state=RANDOM_SEED, verbose=0)

                    # 每次遍历保存实验结果
                    with open(NX_RESULTS_DIR, 'wb') as f:
                        pickle.dump(nx_results, f, protocol=2)

                    with open(TXT_NX_RESULTS_DIR, 'w') as f:
                        json.dump(nx_results, f, indent=4)

            # 保存最终实验结果
            with open(NX_RESULTS_DIR, 'wb') as f:
                pickle.dump(nx_results, f, protocol=2)

            with open(TXT_NX_RESULTS_DIR, 'w+') as f:
                json.dump(nx_results, f, indent=4)

facebook_networks()
#random_networks()


