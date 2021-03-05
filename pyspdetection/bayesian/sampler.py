import numpy as np

from .protocol import SamplerTemplate

class MetropolisSamplerWithDecay():
	
	def __init__(self, temp_init=0.5, half_life_temp=100):
		"""Metropolis accepter with temperature decay. 
		A step is accepted with probability 1 if the likelihood
		has increased. Otherwise, it follows a temperature decay 
		rate of acceptance.
		
		Params
		------------
		temp_init (Float) : Initial temperature of the sampler.
		half_life_temp (Int) : Half-life decay of the temperature.
		
		"""
		super(MetropolisSamplerWithDecay, self).__init__()
		self.temp_init = temp_init
		self.temp_decay = np.log(2)/half_life_temp
		return
	
	def __call__(self, new_val, previous_val, step):
		
		if np.isinf(new_val):
			return False
		
		if new_val>previous_val:
			return True
		
		temp = self.temp_init * np.exp(-self.temp_decay*step)
		
		dval = np.abs(new_val - previous_val)
		prob_accept = np.exp(-(dval/temp))
		
		if np.random.random()<prob_accept:
			return True
		
		return False