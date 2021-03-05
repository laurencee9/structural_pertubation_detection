import networkx as nx
import numpy as np

from .base import GraphModifier

class RemoveRandomEdges(GraphModifier):
    def __init__(self, n_edges=1):
        super(RemoveRandomEdges, self).__init__()
        self.n_edges = n_edges
        return

    def update_graph(self, G):
        for i in range(self.n_edges):
            rdm_edges_id = np.random.choice(range(G.number_of_edges()))
            edges = list(G.edges())
            e = edges[rdm_edges_id]
            G.remove_edge(*e)
        return G

class RemoveRandomNodes(GraphModifier):
    def __init__(self, n_nodes=1):
        super(RemoveRandomNodes, self).__init__()
        self.n_nodes = n_nodes
        return

    def update_graph(self, G):
        rdm_nodes = np.random.choice(G.nodes(), self.n_nodes)

        if nx.is_directed(G):
            for n in rdm_nodes:
                prede = list(G.predecessors(n))
                for j in prede:
                    G.remove_edge(j,n)
                prede = list(G.successors(n))
                for j in prede:
                    G.remove_edge(n,j)
            return G


        for n in rdm_nodes:
            neighbors = list(G.neighbors(n))
            for j in neighbors:
                G.remove_edge(n, j)
        return G


class RemoveRandomDegreeEdges(GraphModifier):
    def __init__(self, n_edges=1):
        super(RemoveRandomDegreeEdges, self).__init__()
        self.n_edges = n_edges
        return

    def update_graph(self, G):
        repeated_nodes = []
        for n in G.nodes:
            d = G.degree(n)
            repeated_nodes.extend([n] * d)

        i = 0
        while i<self.n_edges:
            source = np.random.choice(repeated_nodes)
            neighbors = list(G.neighbors(source))
            if len(neighbors)>0:
                target = np.random.choice(neighbors)
                G.remove_edge(source, target)
                i += 1
            if G.number_of_edges()<1:
                break

        return G


class RemoveRandomDegreeNodes(GraphModifier):
    def __init__(self, n_nodes):
        super(RemoveRandomDegreeNodes, self).__init__()
        self.n_nodes = n_nodes
        return

    def update_graph(self, G):
        repeated_nodes = []
        for n in G.nodes:
            d = G.degree(n)
            repeated_nodes.extend([n] * d)
        for  i in range(self.n_nodes):
            source = np.random.choice(repeated_nodes)
            neighbors = list(G.neighbors(source))
            for target in neighbors:
                G.remove_edge(source, target)
        return G

class RemoveSelectedEdges(GraphModifier):
    def __init__(self, edges):
        super(RemoveSelectedEdges, self).__init__()
        self.edges = edges
        return

    def update_graph(self, G):
        for e in self.edges:
            G.remove_edge(*e)
        return G


class RemoveSelectedNodes(GraphModifier):
    def __init__(self, nodes):
        super(RemoveSelectedNodes, self).__init__()
        self.nodes = nodes
        return

    def update_graph(self, G):
        for n in self.nodes:
            neighbors = list(G.neighbors(n))
            for j in neighbors:
                G.remove_edge(n, j)
        return G

class CascadeEdgeRemoval(GraphModifier):
    """CascadeEdgeRemoval
    Snowball removing of the edges, starting from a seed node. We stop when
    the number of removed edges is reached.
    """
    def __init__(self, n_edges=1, seed_node=0):
        super(CascadeEdgeRemoval, self).__init__()
        self.seed_node = seed_node
        self.n_edges = n_edges
        self.boundary_set = set()
        return

    def update_graph(self, G):
        for neighbor in G.neighbors(self.seed_node):
            self.boundary_set.add(self._order_edge((self.seed_node,neighbor)))
        for i in range(self.n_edges):
            e = np.random.choice(self.boundary_set, 1)[0]
            self._extend_boundary(G, e)
            G.remove_edge(*e)
            self.boundary_set.remove(e)
        return G

    def _order_edge(self, e):
        if e[0] > e[1]:
           e = tuple(reversed(e))
        return e

    def _extend_boundary(self, G, e):
        """add all available edges from both endpoints"""
        for node in e:
            for neighbor in G.neighbors(node):
                self.boundary_set.add(self._order_edge((node,neighbor)))