import numpy as np
from copy import copy

from .protocol import ProposerTemplate


class Proposer(ProposerTemplate):
	"""Propose steps for the sampler. It chooses
	randomly from a list of edges.
	
	Params
	-----------
	edges (list): List of edges for sampling
	
	"""
	def __init__(self, edges):
		super(Proposer, self).__init__()
		self.__edges = copy(edges)
		return
	
	def __call__(self):
		idx = np.random.choice(np.arange(len(self.__edges)))
		return self.__edges[idx]