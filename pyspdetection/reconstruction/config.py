class ReconstructionConfig():

	@classmethod
	def correlation(cls):
		
		cls = cls()
		cls.name = "correlation"
		cls.model_kwargs = {}
		cls.is_directed = False
		return cls

	@classmethod
	def granger(cls):
		
		cls = cls()
		cls.name = "granger"
		cls.model_kwargs = {"lag": 1}
		cls.is_directed = False
		return cls