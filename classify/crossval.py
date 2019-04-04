import random
from sklearn.model_selection import KFold
import numpy as np

class CrossValidation:
  def __init__(self, n_grams, labels, size):
    self.validation_size = size

    self.categories = [[] for _ in range(4)]

    # Sort samples into individual labels
    for i in range(0, len(labels)):
      self.categories[labels[i]].append(n_grams[i])

    # Shuffle the samples in each label category, while doubling the samples for
    # labels in [1, 2]
    labels_to_double = range(1, 3)
    for cat_idx in range(4):
      if cat_idx in labels_to_double:
        self.categories[cat_idx] += self.categories[cat_idx].copy()
      random.shuffle(self.categories[cat_idx])

    self.indices = [
      KFold(
        n_splits=self.validation_size
      ).split(self.categories[cat_idx]) for cat_idx in range(4)
    ]

    # Build the per-class index lists
    self.index_lists = [([], []) for _ in range(4)]
    for cat_idx, cat_indices in enumerate(self.indices):
      for start, end in cat_indices:
        self.index_lists[cat_idx][0].append(start)
        self.index_lists[cat_idx][1].append(end)

  def get_sets(self, k):
    TRAINING_SET = 0
    TEST_SET = 1
    FEATURE_VECTOR = 0
    LABEL = 1
    data_sets = ([], [],)

    for trn_test in [TRAINING_SET, TEST_SET,]:
      for cat_idx in range(4):
        for index in self.index_lists[cat_idx][trn_test][k]:
          data_sets[trn_test].append(
            (self.categories[cat_idx][index], cat_idx,)
          )
      random.shuffle(data_sets[trn_test])
    training_samples = [a_set_tuple[FEATURE_VECTOR] for a_set_tuple in data_sets[TRAINING_SET]]
    training_labels = [a_set_tuple[LABEL] for a_set_tuple in data_sets[TRAINING_SET]]
    test_samples = [a_set_tuple[FEATURE_VECTOR] for a_set_tuple in data_sets[TEST_SET]]
    test_labels = [a_set_tuple[LABEL] for a_set_tuple in data_sets[TEST_SET]]

    return np.array(training_samples), \
      np.array(training_labels), \
      np.array(test_samples), \
      np.array(test_labels)

# vim: set ts=2 sw=2 expandtab:
