import numpy as np
import networkx as nx
from scipy.integrate import odeint

from .base import BaseDynamics

class ThetaDynamics(BaseDynamics):
    
    def __init__(self, sigma=0.1, input_intensity=[0,1], **kwargs):
        super(ThetaDynamics, self).__init__(**kwargs)
        self.sigma = sigma
        self.input_intensity = input_intensity
        return
    
    def generate_x0(self, G):
        return np.random.uniform(0,2*np.pi, size=(G.number_of_nodes(),))
    
    def time_series(self, x0, G, T):
        
        W = nx.to_numpy_array(G)
        N = G.number_of_nodes()

        def ode(x, T, W):
            I = np.random.uniform(self.input_intensity[0], self.input_intensity[1], size=x.shape)
            net = I + self.sigma/N * (W @ (1-np.cos(x)))
            dydt = (1-np.cos(x)) + (1+np.cos(x))*net
            return dydt

        dt = T[1]-T[0]
        x = x0.copy()
        X = np.zeros((N, len(T)))

        for i,t in enumerate(T):
             
            x += ode(x, t, W)*dt
            X[:,i] = x.copy()  
        return X.T


class DiscretizeThetaModel():

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def __call__(self, X):
        Y = 1-np.cos(X)
        X = np.zeros(Y.shape)
        X[Y>self.threshold] = 1
        return X