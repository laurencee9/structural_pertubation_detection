from .base import GNNModel
from .gnn import SISNet, LotkaNet, ThetaNet

import torch
import torch.nn.functional as F

import numpy as np
import copy

device = torch.device("cpu")

class SISPredictor(GNNModel):

	def __init__(self, config):
		super(SISPredictor, self).__init__(in_channels=config.in_channels,
				out_channels=config.out_channels,
				is_directed=config.is_directed)
		self.__config = copy.copy(config)
		self.prepare_net()
		return

	def prepare_net(self):
		self._model = SISNet(self.__config.in_channels, self.__config.out_channels).to(device)

	def loss_fct(self, y_pred, y):
		return F.binary_cross_entropy(y_pred[0], y[0])

	def predict(self, X, W0, t_pertub, show_progress=True, debug=False):

		return self.predict_missing_edges(X, W0, t_pertub, 
                                learning_rates1=self.__config.learning_rates1, 
                                learning_rates2=self.__config.learning_rates2, 
                                n_epochs=self.__config.num_epochs,
                                prior=self.__config.prior, 
                                weight_decay=self.__config.weight_decay, 
                                norm_output=self.__config.norm_output,
                                show_progress=show_progress,
                                debug=debug)



class LotkaPredictor(GNNModel):

    def __init__(self, config):
        super(LotkaPredictor, self).__init__(in_channels=config.in_channels,
                out_channels=config.out_channels,
                is_directed=config.is_directed)
        self.__config = copy.copy(config)
        self.prepare_net()
        return

    def dataset_transform(self, x):
        return x

    def prepare_net(self):
        self._model = LotkaNet(self.__config.in_channels, self.__config.out_channels).to(device)

    def loss_fct(self, y_pred, y):
        return F.l1_loss(y_pred[0], y[0])

    def predict(self, X, W0, t_pertub, show_progress=True):

        return self.predict_missing_edges(X, W0, t_pertub, 
                                learning_rates1=self.__config.learning_rates1, 
                                learning_rates2=self.__config.learning_rates2, 
                                n_epochs=self.__config.num_epochs,
                                prior=self.__config.prior, 
                                weight_decay=self.__config.weight_decay, 
                                norm_output=self.__config.norm_output,
                                show_progress=show_progress,
                                debug=False)


class ThetaPredictor(GNNModel):

	def __init__(self, config):
		super(ThetaPredictor, self).__init__(in_channels=config.in_channels,
				out_channels=config.out_channels,
				is_directed=config.is_directed)
		self.__config = copy.copy(config)
		self.prepare_net()
		return

	def dataset_transform(self, x):
		# To discrete

		return x

	def prepare_net(self):
		self._model = ThetaNet(self.__config.in_channels, self.__config.out_channels).to(device)

	def loss_fct(self, y_pred, y):
		return F.binary_cross_entropy(y_pred[0], y[0])

	def predict(self, X, W0, t_pertub, show_progress=True):

		return self.predict_missing_edges(X, W0, t_pertub, 
                                learning_rates1=self.__config.learning_rates1, 
                                learning_rates2=self.__config.learning_rates2, 
                                n_epochs=self.__config.num_epochs,
                                prior=self.__config.prior, 
                                weight_decay=self.__config.weight_decay, 
                                norm_output=self.__config.norm_output,
                                show_progress=show_progress,
                                debug=False)



