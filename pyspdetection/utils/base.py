import torch
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable


# --------- General ---------
def shuffle_tensor(tensor):
    idx = torch.randperm(tensor.nelement())
    return tensor.view(-1)[idx].view(tensor.size())

def moving_average(y, rho=0.9):
    x = [y[0]]
    for i in range(len(y)):
        x.append(rho*x[-1]+(1-rho)*y[i])
    return x

# --------- Loss ---------
def weighted_bootstrap_mse_loss(y_pred, avr_rdm_loss, y):
    u = (y_pred - y)**2 / avr_rdm_loss
    return torch.mean(u)

def weighted_mse_loss(y_pred, rdm_y_pred, y):
    u = (y_pred - y)**2 / (rdm_y_pred-y)**2
    return torch.mean(u)


# --------- Dataset ---------
def get_random_slice(x, in_channels, out_channels, return_time=False):
    T = int(x.shape[-1])
    t = np.random.randint(0,T-in_channels-out_channels)
    if return_time:
        return x[...,t:t+in_channels], x[...,t+in_channels:t+in_channels+out_channels], t
    else:
        return x[...,t:t+in_channels], x[...,t+in_channels:t+in_channels+out_channels]

def sampling_using_degree(batch_size, data, scaling):
    in_deg = data.edge_index.sum(1) + data.edge_index.sum(0)
    p = in_deg.detach().numpy()[0].copy()**scaling
    p /= np.sum(p)
    train_idx = np.random.choice(range(data.num_nodes), size=batch_size, p=p)
    return train_idx

def plot_persitence(grad, r=1, linspace=100):
    plt.figure(figsize=(3*r,2*r))

    thresholds = np.linspace(0,1,linspace)
    y = [np.sum(grad>t) for t in thresholds]
    plt.plot(thresholds, y, "-", lw=2)
    plt.xlabel("Gradient threshold"), plt.ylabel("Number of removed edges")
    plt.grid()
    return thresholds, y



def isiterable(val):
    if type(val)==str:
        return False
    return isinstance(val, Iterable)