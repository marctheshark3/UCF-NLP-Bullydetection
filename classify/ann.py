import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from . import Metrics


def ann_train_and_evaluate(training_set_samples, training_set_labels, test_set_samples, test_set_labels, xmit_conn):
  """
  Functional unit that accepts training and test sets, creates a
  fully-connected Artificial Neuron Network (ANN) classifier, trains it, and
  evaluates it. The results are communicated via the transmit-only connection
  provided as an argument. This function is meant to be used as the "target"
  of a Python :py:class:multiprocessing.Process instance. Making this
  functional unit a valid target for :py:class:multiprocessing.Process helps
  overcome the TensorFlow/Keras CUDA acceleration limitation where allocated
  GPU memory in a process is never released. Executing this logic in a
  separate process ensures that memory is released after every model run. This
  implementation also works in CPU-only runs.

  :param training_set_samples: Set of row-major samples used to train the ANN
  model.
  :type training_set_samples: numpy.ndarray
  :param training_set_labels: Set of row-major labels associated with each
  training sample.
  :type training_set_labels: numpy.ndarray
  :param test_set_samples: Set of row-major samples used to evaluate the ANN
  model.
  :type test_set_samples: numpy.ndarray
  :param test_set_labels: Set of row-major labels associated with each test
  sample.
  :type test_set_labels: numpy.ndarray
  :param xmit_conn: Transmit only connection object used to send the results
  of the model evaluation in the form of metrics.
  :type xmit_conn: multiprocessing.Connection
  """
  training_sample_count = training_set_samples.shape[0]
  n_gram_count = training_set_samples.shape[1]
  unique_class_count = np.max(np.unique(training_set_labels)) + 1
  one_hot_training_labels = np.zeros((training_sample_count, unique_class_count))
  one_hot_training_labels[np.arange(training_sample_count), training_set_labels] = 1.
  model = keras.Sequential()
  model.add(
    keras.layers.InputLayer(
      input_shape=(n_gram_count,),
      name='input_layer'
    )
  )
  model.add(
    keras.layers.Dense(
      1000,
      activation=tf.nn.sigmoid,
      name='hidden_layer'
    )
  )
  model.add(
    keras.layers.Dense(
      one_hot_training_labels.shape[1],
      activation=tf.nn.sigmoid,
      name='output_layer'
    )
  )
  model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
  model.fit(training_set_samples, one_hot_training_labels, epochs=3)
  one_hot_predictions = model.predict(test_set_samples)
  predictions = np.argmax(one_hot_predictions, axis=1)
  conf_matrix = confusion_matrix(test_set_labels, predictions, labels=[x for x in range(unique_class_count)])
  precision_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
  recall_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
  overall_accuracy = np.sum(np.diag(conf_matrix)) / test_set_samples.shape[0]
  metrics = {
    'confusion_matrix': conf_matrix,
    Metrics.PRECISION: precision_per_class,
    Metrics.RECALL: recall_per_class,
    Metrics.ACCURACY: overall_accuracy
  }
  # test_sample_count = test_set_samples.shape[0]
  # one_hot_test_labels = np.zeros((test_sample_count, unique_class_count))
  # one_hot_test_labels[np.arange(test_sample_count), test_set_labels] = 1.
  # confusion_matrix = np.zeros((unique_class_count, unique_class_count,))
  # true_positive = [0. for _ in range(unique_class_count)]
  # false_negative = [0. for _ in range(unique_class_count)]
  # for prediction_idx, a_prediction in enumerate(predictions[:]):
  #   prediction_idx = np.argmax(a_prediction)
  #   actual_idx = np.argmax(one_hot_test_labels[prediction_idx])
  #   confusion_matrix[actual_idx, prediction_idx] += 1.
  #   if actual_idx == prediction_idx:
  #     true_positive[actual_idx] += 1.
  #   else:
  #     false_negative[actual_idx] += 1.
  # metrics = {
  #   'confusion_matrix': confusion_matrix,
  #   Metrics.RECALL: 
  # }
  xmit_conn.send(metrics)

# vim: set ts=2 sw=2 expandtab:
