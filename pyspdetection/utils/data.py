from pyzoo.datasets import BaseDataset
from pyzoo.datasets.graphs import RealNetwork, TemporalNetworks

import networkx as nx


def get_data(G, perturbation, dynamics, T, t0):
    
    graph_generator = RealNetwork(G.copy())
    modifier = [(t0,perturbation)]
    networks = TemporalNetworks.from_modifiers(modifier, graph_generator)
    dataset = BaseDataset(networks, dynamics)
    Xtrain = dataset.time_series(T, burn=100).T
    G0 = networks(0)
    G1 = networks(T[-1]+100)
    W0 = nx.to_numpy_array(G0)
    W1 = nx.to_numpy_array(G1)
    return (W0,W1), (G0,G1), Xtrain


