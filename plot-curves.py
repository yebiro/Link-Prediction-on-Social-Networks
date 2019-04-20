import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import link_prediction_scores as lp
import pickle
from sklearn import metrics
import numpy as np

RANDOM_SEED = 0

# 绘制ROC曲线和PR曲线
# 打开facebook链路预测结果
fb_results = None
with open('./results/fb-experiment-1-results.pkl', 'rb') as f:
    fb_results = pickle.load(f, encoding='latin1')


##1. 绘制 ROC 曲线
def show_roc_curve(graph_name, frac_hidden, method):
    myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
    results_dict = fb_results['fb-{}-{}-hidden'.format(graph_name, frac_hidden)]
    roc_curve = results_dict[method]['test_roc_curve']
    test_roc = results_dict[method]['test_roc']
    fpr, tpr, _ = roc_curve

    plt.figure(figsize=(12, 9))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
       lw=lw, label='ROC curve (AUC = %0.4f)' % test_roc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳率', fontproperties = myfont)
    plt.ylabel('真阳率', fontproperties = myfont)
    hidden = "%0.0f%%" % ((frac_hidden - 0.35) * 100)
    title = 'ROC Curve:\nFB-{} graph, {} hidden, {}'.format(graph_name, hidden, 'gae')
    #plt.title(title)
    plt.legend(loc="lower right")
    TABLE_RESULTS_DIR = './results/tables/' + 'ROC Facebook-{}, {}'.format(graph_name, method) +'.png'
    plt.savefig(TABLE_RESULTS_DIR)
    plt.show()

#show_roc_curve('combined', 0.5, 'pa')


##2. 绘制 PR 曲线
def show_pr_curve(graph_name, frac_hidden, method):
    myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
    results_dict1 = fb_results['fb-{}-{}-hidden'.format(1912, frac_hidden)]
    pr_curve1 = results_dict1[method]['test_pr_curve']
    test_ap1 = results_dict1[method]['test_ap']
    recall1, precision1, _ = pr_curve1

    results_dict2 = fb_results['fb-{}-{}-hidden'.format(1684, frac_hidden)]
    pr_curve2 = results_dict2[method]['test_pr_curve']
    test_ap2 = results_dict2[method]['test_ap']
    recall2, precision2, _ = pr_curve2

    results_dict3 = fb_results['fb-{}-{}-hidden'.format(686, frac_hidden)]
    pr_curve3 = results_dict3[method]['test_pr_curve']
    test_ap3 = results_dict3[method]['test_ap']
    recall3, precision3, _ = pr_curve3

    plt.figure()
    lw = 2
    plt.plot(recall1, precision1, color='darkorange',
       lw=lw, label='A (AP = %0.4f)' % test_ap1)
    plt.plot(recall2, precision2, color='r',
             lw=lw, label='B (AP = %0.4f)' % test_ap2)
    plt.plot(recall3, precision3, color='b',
             lw=lw, label='C (AP = %0.4f)' % test_ap3)

    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('查全率', fontproperties = myfont)
    plt.ylabel('查准率', fontproperties = myfont)
    hidden = "%0.0f%%" % ((frac_hidden - 0.35) * 100)
    title = 'Precision-Recall Curve '
    plt.title(title)
    plt.legend(loc="lower left")
    TABLE_RESULTS_DIR = './results/tables/' + 'PR Facebook-{}, {}'.format('example', method) +'.png'
    plt.savefig(TABLE_RESULTS_DIR)
    plt.show()

# FB_EGO_USERS = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
# for fb in FB_EGO_USERS:
#     show_pr_curve(fb, 0.15, 'gae_edge_emb')
show_pr_curve('0', 0.15, 'gae_edge_emb')


