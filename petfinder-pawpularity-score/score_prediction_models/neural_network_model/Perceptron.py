class Perceptron:
	def __init__(self, num_features):
		self.num_features = num_features
		self.weights = [0.0 for _ in range(num_features)]
		self.bias = 0.0

	def forward(self, x):
		weighted_sum_z = self.bias
		for i, _ in enumerate(self.weights):
			weighted_sum_z += x[i] * self.weights[i]

		if weighted_sum_z > 0.0:
			prediction = 1
		else:
			prediction = 0

		return prediction

	def update(self, x, true_y):
		prediction = self.forward(x)
		error = true_y - prediction

		# update
		self.bias += error
		for i, _ in enumerate(self.weights):
			self.weights[i] += error * x[i]

		return error

	def train(model, all_x, all_y, epochs):

		for epoch in range(epochs):
			error_count = 0

			for x, y in zip(all_x, all_y):
				error = model.update(x, y)
				error_count += abs(error)
			
			print(f"Epoch {epoch+1} errors {error_count}")
			
			if (error_count == 0):
				break