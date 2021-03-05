import os
import numpy as np
from abc import ABC, abstractmethod


class GraphModifier(ABC):
    
    def __init__(self):
        super(GraphModifier, self).__init__()
        return

    @abstractmethod
    def update_graph(self, G):
        """Return an update graph with a modified structure.
        """
        raise NotImplemented("update_graph must be implemented")

class GraphGenerator(ABC):
    def __init__(self):
        super(GraphGenerator, self).__init__()
        return

    @abstractmethod
    def __call__(self):
        """Generates a graph.
        """
        raise NotImplemented("__call__ method must be implemented")

class GraphEvolution():
    """Simple class that returns the networkx graph at a certain time
    """
    def __init__(self, data):
        self.graphs = [d[1] for d in data]
        self.times  = [d[0] for d in data]
        return

    def __call__(self, time):
        for i,t in enumerate(self.times):
            if t>time:
                return self.graphs[i-1]
        return self.graphs[-1]

class TemporalNetworks():
    """
    """
    def __init__(self):
        self.graphs = None
        self.modifiers = []
        self.events = []
        return

    @classmethod
    def from_graphs(cls, data):
        """Construct the class from temporal graphs data.
        Params
        ---------------
        data: List of tuples : [(t,G), (t1, G1), ...] 
        """
        cls = cls()
        cls.graphs = GraphEvolution(data)
        return cls

    @classmethod
    def from_modifiers(cls, modifiers, graph_generator):
        """
        """
        cls = cls()
        cls.modifiers = modifiers
        data = TemporalNetworks._get_graph_evolution(graph_generator(), modifiers)
        cls.graphs = GraphEvolution(data)
        return cls

    @staticmethod
    def _get_graph_evolution(G, modifiers):
        """
        Params
        ------------
        G : Networkx graph at time t0
        """
        graphs = [(0,G.copy())]
        for mod in modifiers:
            t = mod[0]
            G1 = mod[1].update_graph(G.copy())
            graphs.append((t, G1.copy()))
            G = G1.copy()
        return graphs

    @property
    def perturbations_events(self):
        return self.graphs.times.copy()

    def get_all_graph_states(self):
        gs = self.graphs.graphs
        ts = self.graphs.times
        return [(ts[i], gs[i]) for i in range(len(ts))]

    def __call__(self, t):
        """
        Returns the graph at time t
        """
        if self.graphs == None:
            raise KeyError("The class is not properly initialized.")
        return self.graphs(t)
