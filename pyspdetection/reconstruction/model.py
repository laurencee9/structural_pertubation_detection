
import networkx as nx
import numpy as np
from netrd.reconstruction import *
import copy
from .base import ReconstructionModel
from pyspdetection.utils.granger import TimeGrangerCausality


class CorrelationModel(ReconstructionModel):

	def __init__(self, config):
		super(CorrelationModel, self).__init__()
		assert config.name == "correlation", "The config name is incorrect."
		self.__config = copy.copy(config)

	def setup_model(self):
		self._model = CorrelationMatrix()

	def composition(self, W1_pred, W0):

		# W1_pred between -1 and 1
		W1_pred = np.abs(W1_pred)
		np.fill_diagonal(W1_pred,0)

		# Remove non-existing edges
		W1_pred = np.multiply(W1_pred, W0)
		
		# Average if can be undirected
		if self.__config.is_directed==False:
			W1_pred += W1_pred.T

		# Normalized
		if np.max(W1_pred)>0:
			W1_pred /= np.max(W1_pred)
		else:
			W1_pred = np.zeros_like(W1_pred)

		# dW is the matrix of perturbed edges
		dW = np.abs(W1_pred-W0)
		return dW

	def predict(self, X, W0, t_pertub, show_progress=False):

		return self.predict_missing_edges(X, W0, t_pertub, 
							**self.__config.model_kwargs)
		


class GrangerCausality(ReconstructionModel):

	def __init__(self, config):
		super(GrangerCausality, self).__init__()
		assert config.name == "granger", "The config name is incorrect."

		self.__config = copy.copy(config)

	def setup_model(self):
		self._model = TimeGrangerCausality()

	def composition(self, W1_pred, W0):

		# W1_pred between -infty and infty
		np.fill_diagonal(W1_pred,0)

		# Remove non-existing edges
		W1_pred = np.multiply(W1_pred, W0)
		
		# Norm between 0 and 1
		W1_pred[W1_pred<0] = 0

		# Average if can be undirected
		if self.__config.is_directed==False:
			W1_pred += W1_pred.T

		# Normalized
		if np.max(W1_pred)>0:
			W1_pred /= np.max(W1_pred)
		else:
			W1_pred = np.zeros_like(W1_pred)

		# dW is the matrix of perturbed edges
		dW = np.abs(W1_pred-W0)
		return dW

	def predict(self, X, W0, t_pertub, show_progress=False):

		return self.predict_missing_edges(X, W0, t_pertub, 
							**self.__config.model_kwargs)
		















