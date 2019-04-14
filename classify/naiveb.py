import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

class NaiveB:
	def __init__(self):
		self.model = GaussianNB()
		self.labels = []
		self.true_l = []

	def fit(self, train_d, train_l):
		self.model.fit(train_d, train_l)

	def test(self, test_d, test_l):
		self.true_l = test_l
		self.labels = self.model.predict(test_d)

	def printstats(self):
		print("\nAccuracy")
		print(accuracy_score(self.true_l, self.labels))
		print("\nRecall")
		print(recall_score(self.true_l, self.labels,[0,1,2,3], average = 'macro'))
		print("\nConfusion Matrix")
		print(confusion_matrix(self.true_l, self.labels, [0,1,2,3]))
		print("----------------------")