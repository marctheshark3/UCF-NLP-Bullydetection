import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

# Initialize the classifier
class NaiveB:
	def __init__(self):
		self.model = GaussianNB([0.787, 0.06, 0.06, 0.093])
		self.labels = []
		self.true_l = []
		self.precision = []
		self.recall = []
		self.f1 = []
		self.k = 0

		print("*-----------------------------*")
		print("Initialized Naive Bayes")

# Train the model
	def fit(self, train_d, train_l):
		print("*-----------------------------*")
		print("Training")
		self.model.fit(train_d, train_l)

# Get predictions
	def test(self, test_d, test_l):
		print("*-----------------------------*")
		print("Testing")
		self.true_l = test_l
		self.labels = self.model.predict(test_d)

# Calculate stats. If display == True, print stats as well
	def print_stats(self, display):
		prec = precision_score(self.true_l, self.labels, [0,1,2,3], average = 'macro')
		self.precision.append(prec)

		rec = recall_score(self.true_l, self.labels,[0,1,2,3], average = 'macro')
		self.recall.append(rec)

		f1 = f1_score(self.true_l, self.labels, [0,1,2,3], average = 'macro')
		self.f1.append(f1)

		if display:
			print("*-----------------------------*")
			print("\nPrecision")
			print(prec)
			print("\nRecall")
			print(rec)
			print("\nF1")
			print(f1)

# Print average over all runs
	def print_final_stats(self):
		print("*-----------------------------*")
		print(self.k)
		print("Final Statistics")
		print("\nPrecision")
		print(np.sum(self.precision)/self.k)
		print("\nRecall")
		print(np.sum(self.recall)/self.k)
		print("\nF1")
		print(np.sum(self.f1, axis = 0)/self.k)
		print("*-----------------------------*")		

# Reset the model, increase the count
	def reset(self):
		self.model = GaussianNB([0.787, 0.06, 0.06, 0.093])
		self.k +=1