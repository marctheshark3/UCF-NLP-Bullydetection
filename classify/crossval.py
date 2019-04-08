import random
import numpy as np
from sklearn.model_selection import KFold

# Create an object which stores validation and test indecies
class CrossValidation:
  def __init__(self, n_grams, labels, size):
    self.validation_size = size

    # Separate each n-gram into list of categories
    self.categories = [[], [], [], []]
    for i in range(0, len(labels)):
      self.categories[labels[i]].append(n_grams[i])

    # Double count necessary categories
    double_count_1 = self.categories[1].copy()
    self.categories[1]+= double_count_1

    double_count_2 = self.categories[2].copy()
    self.categories[2]+=double_count_2

    # Get the indecies of training and testing sets
    self.indecies = [[],[],[],[]]
    for category in range(0,4):
      self.indecies[category] = KFold(n_splits=self.validation_size).split(self.categories[category])

    # Separate eacg of the lists into a category with separate lists for training and testing indecies
    self.list = [[[],[]],[[],[]],[[],[]],[[],[]]]
    for category in range(0, 4):
      for start, end in self.indecies[category]:
        self.list[category][0].append(start)
        self.list[category][1].append(end)


  # Return the training and testing sets/labels
  def get_sets(self, k):
    test_set = []
    train_set = []
    test_labels = []
    train_labels = []

    for category in range(0, 4):
      for index in self.list[category][1][k]:
        test_set.append(self.categories[category][index])
        test_labels.append(category)

    for category in range(0, 4):
      for index in self.list[category][0][k]:
        train_set.append(self.categories[0][index])
        train_labels.append(category)

    #Shuffle the data
    random.seed(30)
    random.shuffle(train_set)
    random.shuffle(test_set)
    random.seed(30)
    random.shuffle(train_labels)
    random.shuffle(test_labels)

    return np.array(train_set), np.array(train_labels), np.array(test_set), np.array(test_labels)

# vim: set ts=2 sw=2 expandtab:

