import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# Documentation: https://github.com/kimiyoung/planetoid
    # x: the feature vectors of the labeled training instances
    # tx: the feature vectors of the test instances
    # allx: the feature vectors of both labeled and unlabeled training instances (a superset of x)
    # graph: a dict in the format {index: [index_of_neighbor_nodes]}
def load_data(dataset, datatype):
    if datatype == 1:
        graph = []
        name = 'edgelist'
        with open("data/out.{}.{}".format(dataset, name), 'rb') as f:
            graph = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=str, encoding='latin1', data=(('weight', float),))

        print(type(graph))
        adj = nx.adjacency_matrix(graph)
        #print(adj)
        # features = sp.vstack((graph)).tolil()
        # print(1+features)
        features = adj

        # nx.draw(graph)
        # plt.savefig("data/graph.png")
        # plt.show()

        return adj, features
    else:
        # load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        for i in range(len(names)):
            with open("datacontent/{}.txt".format(names[i]), 'w+') as data:
                if names[i] == 'x':
                    print(x, file=data)
                if names[i] == 'tx':
                    print(tx, file=data)
                if names[i] == 'allx':
                    print(allx, file=data)
                if names[i] == 'graph':
                    print(graph, file=data)
                data.close()

        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended

        features = sp.vstack((allx, tx)).tolil()
        with open("datacontent/features1.txt", 'w+') as features1:
            print(features, file=features1)
            features1.close()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        with open("datacontent/features.txt", 'w+') as data1:
            print(features, file=data1)
            data1.close()
        with open("datacontent/adj.txt", 'w+') as data2:
            print(adj, file=data2)
            data2.close()
        with open("datacontent/test_idx_reorder.txt", 'w+') as data3:
            print(test_idx_reorder, file=data3)
            data3.close()
        with open("datacontent/test_idx_range.txt", 'w+') as data4:
            print(test_idx_range, file=data4)
            data4.close()

        return adj, features


