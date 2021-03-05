import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn.inits import reset
from .protocol import LikelihoodComputerTemplate

import numpy as np

class SISLikelihoodConv(MessagePassing):
    """Compute the number of active neighbors
    """
    def __init__(self, **kwargs):
        super(SISLikelihoodConv, self).__init__(aggr='add', **kwargs)

    def forward(self, x, edge_index):
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        # print(edge_index.shape, x.shape)
        if edge_index.shape[1]==0:
            return torch.zeros_like(x)
        mi_t = self.propagate(edge_index, x=x)
        return mi_t

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class LikelihoodComputer(LikelihoodComputerTemplate):
    """
    Params
    -------------
    sis_p (float): Probability of infection.
    sis_q (float): Probability of recovery.
    approx_on_steps (int): Maximum number of steps to use for inference
    """
    def __init__(self, sis_p, sis_q, approx_on_steps=None):
        super(LikelihoodComputer, self).__init__()
        self.sis_p = sis_p
        self.sis_q = sis_q
        self.conv = SISLikelihoodConv()
        self.approx_on_steps = approx_on_steps
        return
    
    def __call__(self, X, W, subset=None):
        """Compute the likelihood of a configuration
        under the SIS model.
    
        Returns
        ----------------
        
        log(P(X|W,p,q))
        
        """
        if self.approx_on_steps is not None:
            t1 = X.shape[1]-self.approx_on_steps
            if t1>0:
                t = np.random.randint(0,t1)
                X = X[:,t1:t1+self.approx_on_steps]

        x = torch.from_numpy(X)
        edge_index = torch.from_numpy(np.argwhere(W).T)
        
        mi_t = self.conv(x, edge_index)
        inactive_index = x==0
        p = torch.zeros_like(x) + (1-self.sis_q)
        p[inactive_index] = (1-(1-self.sis_p)**mi_t[inactive_index])
        
        # p gives the probability that next time is 1
        x_next_time = torch.roll(x, -1, dims=1)
        inactive_index = torch.ByteTensor((1-x_next_time).byte())
        p[inactive_index] = 1-p[inactive_index]
        if subset is None:
            return float(torch.log(p)[:,:-1].sum().detach())
        else:
            return float(torch.log(p)[subset,:-1].sum().detach())

class LikelihoodWithFlipComputer(LikelihoodComputerTemplate):
    """
    Params
    -------------
    sis_p (float): Probability of infection.
    sis_q (float): Probability of recovery.
    approx_on_steps (int): Maximum number of steps to use for inference
    """
    def __init__(self, sis_p, sis_q, flip_prob, approx_on_steps=None):
        super(LikelihoodWithFlipComputer, self).__init__()
        self.sis_p = sis_p
        self.sis_q = sis_q
        self.flip_prob = flip_prob
        self.conv = SISLikelihoodConv()
        self.approx_on_steps = approx_on_steps
        return
    
    def __call__(self, X, W, subset=None):
        """Compute the likelihood of a configuration
        under the SIS model.
    
        Returns
        ----------------
        
        log(P(X|W,p,q))
        
        """
        if self.approx_on_steps is not None:
            t1 = X.shape[1]-self.approx_on_steps
            if t1>0:
                t = np.random.randint(0,t1)
                X = X[:,t1:t1+self.approx_on_steps]

        x = torch.from_numpy(X)
        x_next = torch.roll(x, -1, dims=1)

        edge_index = torch.from_numpy(np.argwhere(W).T)
        li_t = self.conv(x, edge_index)

        k = np.expand_dims(np.sum(W, axis=0), axis=-1)
        k = k.repeat(li_t.shape[-1], axis=-1)
        k = torch.from_numpy(k)
        u = self.flip_prob+(1-self.flip_prob)*(1-self.sis_p)
        v = 1-self.flip_prob  + (self.flip_prob*(1-self.sis_p))

        Fkl = (u**li_t) * (v**(k-li_t))

        r = self.flip_prob
        p = self.sis_p
        q = self.sis_q

        # 1|1
        A = (1-r)*((1-q)*(1-r)+q*r)+ r*(1-r)*(1-Fkl)+r*r*Fkl

        # 1|0, u=1, v=0
        B = r*(q*r+(1-q)*(1-r)) + (1-r)*(r*Fkl+(1-r)*(1-Fkl))

        # 0|1, u=0, v=1
        C = (1-r)*(q*(1-r)+(1-q)*r)+r*(r*(1-Fkl)+(1-r)*Fkl)

        # 0|0, u=0, v=0
        D = r*(q*(1-r)+(1-q)*r) + (1-r)*(r*(1-Fkl)+(1-r)*Fkl)

        p = x*x_next*A + (1-x)*x_next*B + C*x*(1-x_next) + D*(1-x)*(1-x_next)

        
        if subset is None:
            return float(torch.log(p)[:,:-1].sum().detach())
        else:
            return float(torch.log(p)[subset,:-1].sum().detach())

    
    