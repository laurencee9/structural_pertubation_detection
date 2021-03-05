from .base import GraphGenerator

import networkx as nx

class ErdosReyniNetworks(GraphGenerator):

    def __init__(self, N, density, directed=False, **kwargs):
        super(ErdosReyniNetworks, self).__init__(**kwargs)
        self.number_of_nodes = N
        self.density = density
        self.directed = directed
        return

    def __call__(self):
        return nx.erdos_renyi_graph(self.number_of_nodes, self.density, directed=self.directed)

class CycleNetworks(GraphGenerator):

    def __init__(self, N, **kwargs):
        super(CycleNetworks, self).__init__(**kwargs)
        self.number_of_nodes = N

    def __call__(self):
        return nx.cycle_graph(self.number_of_nodes)


class RealNetwork(GraphGenerator):
    def __init__(self, G, **kwargs):
        super(RealNetwork, self).__init__(**kwargs)
        self.number_of_nodes = G.number_of_nodes()
        self.G = G.copy()

    def __call__(self):
        return self.G


class StarNetworks(GraphGenerator):
    def __init__(self, N, **kwargs):
        super(StarNetworks, self).__init__(**kwargs)
        self.number_of_nodes = N

    def __call__(self):
        return nx.star_graph(self.number_of_nodes-1)

class RegularNetworks(GraphGenerator):
    def __init__(self, N, degree, **kwargs):
        super(RegularNetworks, self).__init__(**kwargs)
        self.number_of_nodes = N
        self.degree = degree

    def __call__(self):
        return nx.random_regular_graph(self.degree, self.number_of_nodes)


class BarabasiAlbertNetworks(GraphGenerator):

    def __init__(self, N, m, **kwargs):
        super(BarabasiAlbertNetworks, self).__init__(**kwargs)
        self.number_of_nodes = N
        self.number_of_new_edges = m
        return

    def __call__(self):
        return nx.barabasi_albert_graph(self.number_of_nodes,
                                        self.number_of_new_edges)
