
import networkx as nx
import numpy as np
from netrd.reconstruction import *

from abc import ABC, abstractmethod

class ReconstructionModel(ABC):

	def __init__(self):
		self.setup_model()

	@abstractmethod
	def setup_model(self):
		raise NotImplemented("setup_model must be implemented")

	@abstractmethod
	def composition(self, W1_pred, W0):
		raise NotImplemented("composition must be implemented")
			
	def predict_missing_edges(self, X, W0, t_pertub, **kwargs):
		G1_pred = self._model.fit(X[:,t_pertub:], **kwargs)
		W1_pred = nx.to_numpy_array(G1_pred)
		dW = self.composition(W1_pred, W0)
		return dW
