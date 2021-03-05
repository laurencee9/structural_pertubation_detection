import numpy as np

from .base import BaseDynamics

class SISDynamics(BaseDynamics):
    def __init__(self, p, q, self_activation=0):
        """__init__
        :param p: probability of infection
        :param q: probability of recovery
        Susceptible node = 0
        Infected node = 1
        """
        super(SISDynamics, self).__init__()
        self.p = p
        self.q = q
        self.self_activation = self_activation
        self.infected_node_set = set()

    def time_series(self, x0, G, T):
        """time_series generates a sequence of states for each time
        :param x0: Initial state as a vector
        :param G: Networkx graph
        :param T: time vector
        """
        #initialize
        for node in range(len(x0)):
            if x0[node] == 1:
                self.infected_node_set.add(node)
        X = []
        dt = T[1]-T[0]
        for t in T:
            self._step(G)
            X.append(self._get_state(G))
            
        self.infected_node_set = set()
        return np.array(X)

    def generate_x0(self, G):
        N = G.number_of_nodes()
        return np.random.randint(0,2, size=(N,))

    def _step(self, G):
        """_step realize a time step of the process
        """
        new_infected_node_set = self.infected_node_set.copy()
        #look for new infections
        for node in self.infected_node_set:
            #try to infect neighbors
            for neighbor in G.neighbors(node):
                if np.random.random() < self.p:
                    new_infected_node_set.add(neighbor)

        #look for recuperations
        for node in self.infected_node_set:
            #try to recuperate
            if np.random.random() < self.q:
                new_infected_node_set.remove(node)
        #set new infected nodes
        self.infected_node_set = new_infected_node_set

    def _get_state(self, G):
        """_get_state returns the current state"""
        x = np.zeros(len(G))
        for node in self.infected_node_set:
            x[node] = 1

        # Random activation
        if self.self_activation>0:
            rdm_act = np.random.choice([0,1], size=len(x), p=[1-self.self_activation, self.self_activation])
            x = np.minimum(x+rdm_act, 1)
        return x