
import numpy as np
import networkx as nx

from .manager import BayesianManager
from tqdm import tqdm
import copy

class BayesianSamplingModel():

	def __init__(self, config):

		assert config.sis_p is not None, """You most set the sis parameters. 
											Use `set_sis` in the config files."""
		self.__config = copy.copy(config)


	def predict(self, X, W0, t_pertub, show_progress=True, debug=False):

		pbar = None
		if show_progress:
			pbar = tqdm(total=self.__config.num_steps)

		sampler_config = self.get_expert_config("SAMPLER")
		computer_config = self.get_expert_config("COMPUTER")
		proposer_config = self.get_expert_config("PROPOSER")
		
		manager = BayesianManager(X[:,t_pertub:], W0, sampler_config, proposer_config, computer_config)

		for step, _ in manager:
			if step> self.__config.burning_steps:
				break

		manager = BayesianManager(X[:,t_pertub:], manager.W.copy(), sampler_config, proposer_config, computer_config)
		
		for step, _ in manager:
			if step>self.__config.num_steps:
				break
			if pbar is not None:
				pbar.update(1)
		if pbar is not None:
			pbar.close()

		dW = manager.compute_mmse()
		if debug:
			return dW, manager.results, manager
		return dW

	def get_expert_config(self, key):
		attr = self.__config.__dict__
		return {a.split(key+"_")[1]: attr[a] for a in list(attr.keys()) if a.startswith(key+"_")}