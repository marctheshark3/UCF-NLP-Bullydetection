import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

# Initialize the classifier
class NaiveB:
	def __init__(self):
		self.model = GaussianNB()
		self.labels = []
		self.true_l = []
		self.accuracy = []
		self.recall = []
		self.matrix = []
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
		acc = accuracy_score(self.true_l, self.labels)
		self.accuracy.append(acc)
		rec = recall_score(self.true_l, self.labels,[0,1,2,3], average = 'macro')
		self.recall.append(rec)
		matrix = confusion_matrix(self.true_l, self.labels, [0,1,2,3])
		self.matrix.append(matrix)
		if display:
			print("*-----------------------------*")
			print("\nAccuracy")
			print(acc)
			print("\nRecall")
			print(rec)
			print("\nConfusion Matrix")
			print(matrix)

# Print average over all runs
	def print_final_stats(self):
		print("*-----------------------------*")
		print("Final Statistics")
		print("\nAccuracy")
		print(np.sum(self.accuracy)/self.k)
		print("\nRecall")
		print(np.sum(self.recall)/self.k)
		print("\nConfusion Matrix")
		print(np.sum(self.matrix, axis = 0)/self.k)
		print("*-----------------------------*")		

# Reset the model, increase the count
	def reset(self):
		self.model = GaussianNB()
		self.k +=1