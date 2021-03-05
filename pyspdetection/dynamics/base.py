import os
import numpy as np

from abc import ABC, abstractmethod

class BaseDynamics(ABC):

	def __init__(self):
		super(BaseDynamics, self).__init__()
		return

	@abstractmethod
	def time_series(self, x0, G, T):
		raise NotImplementedError("time_series must be implemented.")

	def generate_x0(self, G):
		raise NotImplementedError("generate_x0 must be implemented.")