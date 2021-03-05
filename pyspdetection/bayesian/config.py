class BayesConfig():

	@classmethod
	def default(cls):
		
		cls = cls()

		cls.num_steps = 2500

		cls.SAMPLER_temp_init = 1
		cls.SAMPLER_half_life_temp = 999999999999999

		cls.COMPUTER_approx_on_steps = None

		cls.burning_steps = 500
		cls.sis_p = None
		cls.sis_q = None
		cls.flip_prob = None

		return cls

	@classmethod
	def paper(cls):
		
		cls = cls()

		cls.num_steps = 2000

		cls.SAMPLER_temp_init = 2
		cls.SAMPLER_half_life_temp = 500

		cls.COMPUTER_approx_on_steps = None

		cls.burning_steps = 0
		cls.sis_p = None
		cls.sis_q = None
		cls.flip_prob = None

		return cls

	def set_sis(self, p, q, flip_prob=None):
		self.sis_p = p
		self.sis_q = q
		self.COMPUTER_sis_p = p
		self.COMPUTER_sis_q = q
		if flip_prob is not None:
			self.flip_prob = flip_prob
			self.COMPUTER_flip_prob = flip_prob