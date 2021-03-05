import numpy as np
import networkx as nx
from scipy.integrate import odeint

from .base import BaseDynamics

class LotkaVolterraDynamics(BaseDynamics):
    
    def __init__(self, intra_competition=0.1, alpha=0.75, beta=1, **kwargs):
        super(LotkaVolterraDynamics, self).__init__(**kwargs)
        self.intra_competition = intra_competition
        self.alpha = alpha
        self.beta = beta
        return
    
    def generate_x0(self, G):
        return np.random.uniform(0,1, size=(G.number_of_nodes(),))
    
    def time_series(self, x0, G, T):
        
        W = nx.to_numpy_array(G)

        def ode(x, T, W):
            
            down = W*self.alpha
            up   = W.T*self.beta
            
            dydt = np.array(x) * np.array((down @ np.array(x)))
            dydt -= np.array(x) * np.array((up @ np.array(x)))
            
            interaction = (np.array(np.sum(up, axis=1))-np.array(np.sum(down, axis=1)))
            dydt +=  x * self.intra_competition * interaction
            return dydt
        
        X = odeint(ode, x0, T, args=(W.copy(), ))

        return X 