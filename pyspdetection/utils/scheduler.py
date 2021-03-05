
class LearningRateScheduler():

	def __init__(self, learning_rates):
		self.learning_rates = learning_rates

	def __call__(self, epoch):
		gt_lr = 1e-5
		for t, lr in self.learning_rates:
			if t<=epoch:
				gt_lr = lr
			else:
				return gt_lr

		return gt_lr


