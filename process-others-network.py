import networkx as nx
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from math import log



##################################
######### READ EDGE LIST #########
##################################
NetWorks=['twitter', 'gplus', 'hamster', 'advogato']
#NetWorks=['gplus']

for network in NetWorks:
    # Read edge-list
    print('')
    print('Reading {} edgelist'.format(network))
    network_edges_dir = './data/{}/{}.txt'.format(network,network)

    # Parse edgelist into undirected graph
    if network in ['hamster', 'jazz', 'karate']:
        with open(network_edges_dir, 'rb')as edges_f:
            network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.Graph(), encoding='latin1',
                                         data=(('weight', float),))
        adj = nx.adjacency_matrix(network_g)

        # Save adjacency matrix
        with open('./data/{}/{}-adj.pkl'.format(network,network), 'wb') as f:
            pickle.dump(adj, f)

    else:
        with open(network_edges_dir, 'rb')as edges_f:
            network_g = nx.read_edgelist(edges_f, nodetype=int, create_using=nx.DiGraph(), encoding='latin1',
                                         data=(('weight', float),))
        print('Num. weakly connected components: ', nx.number_weakly_connected_components(network_g))

        # print('Saving adjacency matrix')

        # Get adjacency matrix
        adj = nx.adjacency_matrix(network_g)

        # Save adjacency matrix
        with open('./data/{}/{}-adj.pkl'.format(network, network), 'wb') as f:
            pickle.dump(adj, f)








