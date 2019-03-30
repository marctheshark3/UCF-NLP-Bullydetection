import random
from sklearn.cross_validation import KFold

class Cross_validation:
	def __init__(self, n_grams, labels, size):
		self.validation_size = size

		self.categories = [[], [], [], []]

		for i in range(0, len(labels)):
			self.categories[labels[i]].append(n_grams[i])

		double_count_1 = self.categories[1].copy()
		self.categories[1]+= double_count_1

		double_count_2 = self.categories[2].copy()
		self.categories[2]+=double_count_2

		random.shuffle(self.categories[0])
		random.shuffle(self.categories[1])
		random.shuffle(self.categories[2])
		random.shuffle(self.categories[3])

		self.indecies1 = KFold(len(self.categories[0]), n_folds=self.validation_size)
		self.indecies2 = KFold(len(self.categories[1]), n_folds=self.validation_size)
		self.indecies3 = KFold(len(self.categories[2]), n_folds=self.validation_size)
		self.indecies4 = KFold(len(self.categories[3]), n_folds=self.validation_size)

		self.list1 = [[],[]]
		self.list2 = [[],[]]
		self.list3 = [[],[]]
		self.list4 = [[],[]]

		for start, end in self.indecies1:
			self.list1[0].append(start)
			self.list1[1].append(end)

		for start, end in self.indecies2:
			self.list2[0].append(start)
			self.list2[1].append(end)

		for start, end in self.indecies3:
			self.list3[0].append(start)
			self.list3[1].append(end)

		for start, end in self.indecies4:
			self.list4[0].append(start)
			self.list4[1].append(end)

	def get_sets(self, k):
		test_set = []
		train_set = []
		test_labels = []
		train_labels = []

		for index in self.list1[1][k]:
			test_set.append(self.categories[0][index])
			test_labels.append(0)

		for index in self.list2[1][k]:
			test_set.append(self.categories[1][index])
			test_labels.append(1)

		for index in self.list3[1][k]:
			test_set.append(self.categories[2][index])
			test_labels.append(2)

		for index in self.list4[1][k]:
			test_set.append(self.categories[3][index])
			test_labels.append(3)

		for index in self.list1[0][k]:
			train_set.append(self.categories[0][index])
			train_labels.append(0)

		for index in self.list2[0][k]:
			train_set.append(self.categories[1][index])
			train_labels.append(1)

		for index in self.list3[0][k]:
			train_set.append(self.categories[2][index])
			train_labels.append(2)

		for index in self.list4[0][k]:
			train_set.append(self.categories[3][index])
			train_labels.append(3)

		return train_set, train_labels, test_set, test_labels






		


