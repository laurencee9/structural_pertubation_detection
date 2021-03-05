
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pyspdetection.gnn.dataset import TorchDataset, SequentialSampler
from pyspdetection.utils import shuffle_tensor, weighted_bootstrap_mse_loss, get_random_slice, sampling_using_degree
from pyspdetection.utils.scheduler import LearningRateScheduler
from pyspdetection.utils import RAdam

import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

device = torch.device("cpu")

class GNNModel(ABC):

    def __init__(self, in_channels, out_channels, is_directed=False):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_directed = is_directed

    def dataset_transform(self, x):
        return x

    @abstractmethod
    def loss_fct(self, y_pred, y):
        raise NotImplemented("Model::loss_fct must be implemented")

    def optimization_step(self, optimizers, data, t_pertub, train_idx=None):
        """Single training step with optimization."""
        self._model.train()
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        x, y, t = get_random_slice(data.x, 
            self._model.in_channels, 
            self._model.out_channels, 
            return_time=True)

        p = torch.sigmoid(data.p)
        if t>=t_pertub:
            y_pred = self._model(x, data.adj-data.adj*p)
            wd = torch.sum(p*data.adj)
        else:
            y_pred = self._model(x, data.adj)
            wd = 0

        if train_idx is None:
            loss = self.loss_fct(y_pred, y)
        else:
            loss = self.loss_fct(y_pred[..., train_idx], y[..., train_idx])

        
        loss.backward()
        with torch.no_grad():
            if data.p.grad is not None:
                ind = np.diag_indices(data.p.shape[0])
                data.p.grad[ind[0], ind[1]] = 0

        optimizers[0].step()
        optimizers[1].step()
        with torch.no_grad():
            u  = torch.sum(p.data)

        return loss.data, u, wd

    def predict_missing_edges(self, X, W0, x_index_perturb, 
                                learning_rates1,
                                learning_rates2, 
                                prior=1e-2, 
                                weight_decay=1e-4, 
                                n_epochs=100,
                                show_progress=True,
                                norm_output=True,
                                debug=False):
        """
    
        prior : Prior probability that an edge has been removed.

        """

        scheduler1 = LearningRateScheduler(learning_rates1)
        scheduler2 = LearningRateScheduler(learning_rates2)
        X = self.dataset_transform(X)

        # Create dataset
        dataset = TorchDataset(W0, X)
        data = dataset[0]   
        data.x = data.x.to(device)
        data.adj = data.adj.to(device)

        # Create perturbation variable
        warm_perturbation = - np.log(1/prior - 1)
        perturbation = torch.zeros_like(data.adj) + warm_perturbation
        data.p = Variable(perturbation.float(), requires_grad=True)

        optimizer1 = RAdam(self._model.parameters(), 
                                lr=scheduler1(1), 
                                weight_decay=0.0)

        optimizer2 = RAdam([data.p,], 
                            lr=scheduler2(1), 
                            weight_decay=weight_decay)
        losses = []

        if show_progress:
            iterator = tqdm(range(1,n_epochs+1))
        else:
            iterator = range(1, n_epochs+1)

        for epoch in iterator:

            # Update learning rate
            for g in optimizer1.param_groups:
                g['lr'] = scheduler1(epoch)

            for g in optimizer2.param_groups:
                g['lr'] = scheduler2(epoch)

            loss_value = self.optimization_step([optimizer1, optimizer2], data, x_index_perturb)
            losses.append(loss_value)

        perturbation = torch.sigmoid(data.p).detach().numpy()[0]

        if debug:
            return perturbation, losses

        if self.is_directed == False:
            perturbation = (perturbation+perturbation.T)/2
        else:
            perturbation = perturbation

        # For aesthetic 
        perturbation[W0<1] = 0
        perturbation[W0>0] -= np.min(perturbation[W0>0])
        
        if norm_output:
            perturbation[W0>0] /= np.max(perturbation)

        return perturbation


