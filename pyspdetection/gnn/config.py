class GNNConfig():

	@classmethod
	def SISNet(cls):
		
		cls = cls()
		
		# -- Data params ---
		cls.in_channels = 1
		cls.out_channels = 1
		cls.is_directed = False
		
		# --- Training ---
		cls.num_epochs = 10000
		cls.prior = 1e-2
		cls.weight_decay = 0.0
		cls.norm_output = True

		cls.learning_rates1 = [(0,1e-3)]
		cls.learning_rates2 = [(0,1e-3)]

		return cls

	@classmethod
	def LotkaNet(cls):
		
		cls = cls()
		
		# -- Data params ---
		cls.in_channels = 30
		cls.out_channels = 1
		cls.is_directed = True
		
		# --- Training ---
		cls.num_epochs = 10000
		cls.prior = 1e-2
		cls.weight_decay = 0.0
		cls.norm_output = True

		cls.learning_rates1 = [(0,1e-3)]
		cls.learning_rates2 = [(0,1e-3)]

		return cls

	@classmethod
	def LotkaNetV2(cls):
		
		cls = cls()
		
		# -- Data params ---
		cls.in_channels = 1
		cls.out_channels = 1
		cls.is_directed = True
		
		# --- Training ---
		cls.num_epochs = 5000
		cls.prior = 1e-2
		cls.weight_decay = 0.0
		cls.norm_output = True

		cls.learning_rates1 = [(0,1e-3), (1000,1e-4)]
		cls.learning_rates2 = [(0,1e-2)]
		return cls

	@classmethod
	def ThetaNet(cls):
		
		cls = cls()
		
		# -- Data params ---
		cls.in_channels = 20
		cls.out_channels = 1
		cls.is_directed = True
		
		# --- Training ---
		cls.num_epochs = 10000
		cls.prior = 1e-2
		cls.weight_decay = 0.0
		cls.norm_output = True

		cls.learning_rates1 = [(0,1e-2), (250,1e-3), (1000,1e-4)]
		cls.learning_rates2 = [(0,1e-8), (500,1e-2), (1000,1e-3)]

		return cls

