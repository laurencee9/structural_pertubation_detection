
from abc import ABC, abstractmethod

class LikelihoodComputerTemplate(ABC):

	@abstractmethod
	def __call__(self, X, W):
		"""Should return the log likelihood of 
		observing X knowing W: 
		
		Params
		-------------

		X (np.array(N, T)): Time series
		W (np.array(N, N)): Adjacancy matrix

		Returns
		-------------
		log(P(X|W))

		"""
		return


class ProposerTemplate(ABC):

	@abstractmethod
	def __call__(self):
		"""Should return an edge
		as a proposition
		
		Returns
		------------

		edge: [source, target]
		"""
		return


class SamplerTemplate(ABC):

	@abstractmethod
	def __call__(self, new_val, previous_val, step):
		"""Should return if we should accept the move
		that moves the energy of the system from previous_val
		to new_val. 

		Params
		--------------
		new_val (float): New energy if accepted
		prevous_val (float): Current energy state
		step (int): Number of sampling steps so far

		Returns
		---------------
		Bool : Should accept the move
		"""
		return
