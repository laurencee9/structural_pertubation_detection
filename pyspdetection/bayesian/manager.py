import numpy as np
from copy import copy
import pandas as pd
from collections import defaultdict

from .likelihood import *
from .proposer import *
from .sampler import *


class BayesianManager():
    """
    """
    def __init__(self, X, W0, 
                sampler_config, 
                proposer_config, 
                computer_config):

        self.X = X.copy()
        self.W = W0.copy()
        self.setup(sampler_config, proposer_config, computer_config)
        return
    
    def setup(self, sampler_config, proposer_config, computer_config):
        
        existing_edges = np.argwhere(self.W)
        self.proposer = Proposer(existing_edges, **proposer_config)
        self.judge = MetropolisSamplerWithDecay(**sampler_config)
        
        if "flip_prob" in computer_config:
            print("Using flip")
            self.computer = LikelihoodWithFlipComputer(**computer_config)
        else:
            self.computer = LikelihoodComputer(**computer_config)

        self.log_likelihoog = self.compute_likelihood(None)
        self.results = pd.DataFrame(columns=["edge", "log_likelihood", "is_accepted", "t"])
        
    def should_accept_move(self, edge):
        new_log_likelihood = self.compute_likelihood(edge)
        return self.judge(new_log_likelihood, self.log_likelihoog, self.t), new_log_likelihood

    def compute_mmse(self):

        grads = np.zeros_like(self.W)
  
        remove_edges = defaultdict(int)
        can_compile_edge = defaultdict(bool)

        if len(self.results)==0:
            return grads

        t0 = self.results.head().t[0]
        
        for index, col in self.results.iterrows():
            edge = col.edge
            if edge[0]>edge[1]:
                edge = (edge[1], edge[0])
            else:
                edge = (edge[0], edge[1])

            t = col.t
            dt = t - t0 
            for e in list(remove_edges.keys()):
                if can_compile_edge[e]:
                    remove_edges[e] += dt
            # If we have never seen this edge
            if remove_edges[edge] == 0:
                remove_edges[edge] = 0
                can_compile_edge[edge] = bool(1-can_compile_edge[edge])
            else:
                can_compile_edge[edge] = bool(1-can_compile_edge[edge])
            t0 = t

        if len(remove_edges)==1:
            for e in remove_edges:
                remove_edges[e] = 1
  
        for e in list(remove_edges.keys()):
            grads[e[0], e[1]] += remove_edges[e]
            grads[e[1], e[0]] += remove_edges[e]

        grads /= np.max(grads)
        return grads
    
    def compute_likelihood(self, edge):
        """Compute likelihood of new configuration
        if we flip the edge in argument. 
        """
        W1 = self.W.copy()
        if edge is not None:
            val = 1-W1[edge[0], edge[1]]
            W1[edge[0], edge[1]] = val
            W1[edge[1], edge[0]] = val
        return self.computer(self.X, W1)
    
    def add_state(self, edge, log_likelihood, is_accepted):
        
        self.results = self.results.append({"edge": edge,
                             "log_likelihood":log_likelihood,
                            "is_accepted": is_accepted,
                            "t": self.t}, ignore_index=True)
        return
        
    def __iter__(self):
        self.t = 0
        return self
        
    def __next__(self):
        
        edge = self.proposer()
        move_is_accepted, new_log_likelihood = self.should_accept_move(edge)
        
        if move_is_accepted:
            self.update_graph(edge)
            self.add_state(edge, new_log_likelihood, move_is_accepted)
            self.log_likelihoog = new_log_likelihood
            
        self.t += 1
        return self.t, self.log_likelihoog
            
    def update_graph(self, edge):
        val = 1-self.W[edge[0], edge[1]]
        self.W[edge[0], edge[1]] = val
        self.W[edge[1], edge[0]] = val
        return
        
        
        
        