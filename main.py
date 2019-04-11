import sys
import os
import multiprocessing as mp
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from preproc import translate_file
from featrep import encode_tweets
from classify import CrossValidation, ann_train_and_evaluate, get_svm_metrics
from sklearn import svm

def main(input_file_name, options):
  """
  Top-level program logic. This is an example of the steps executed in series.
  First, the raw data set is pre-processed (with :py:func:translate_file), then
  the pre-processed set is encoded in to feature representations (with
  :py:func:encode_tweets).

  :param str input_file_name: Name of the raw data file to process.
  """
  (name_no_ext, file_ext) = os.path.splitext(input_file_name)
  preproc_file_name = '{:s}-preproc{:s}'.format(name_no_ext, file_ext)
  preproc_regen = True

  try:
    input_file_info = os.stat(input_file_name)
    preproc_file_info = os.stat(preproc_file_name)
    preproc_regen = (preproc_file_info.st_mtime < input_file_info.st_mtime)
  except FileNotFoundError as err:
    if err.filename == input_file_name:
      raise err

  if preproc_regen:
    # Open both the input and output files, and do the transformation. We can add
    # logic here later where, should the `preproc` file already exist and bear a
    # newer timestamp than the original, we would skip the next three (3) lines.
    print('Generating pre-processed file:{:s}'.format(datetime.now().isoformat(sep=' ')))
    with open(input_file_name, "r") as input_file:
      with open(preproc_file_name, "w") as output_file:
        translate_file(input_file, output_file)
    print('Done generating pre-processed file:{:s}'.format(datetime.now().isoformat(sep=' ')))
  else:
    print('No need to re-generate pre-processed file.')

  # Read in the pre-processed data set
  text_data_set = pd.read_csv(preproc_file_name, dtype={'tweet': str, 'label': np.int8})

  feature_representations = {
    '3-gram': encode_tweets(text_data_set['tweet'], [3,]),
    '4-gram': encode_tweets(text_data_set['tweet'], [4,]),
    '3 and 4-gram': encode_tweets(text_data_set['tweet'], [3, 4,]),
  }

  for a_feat_rep_id in options.ngrams.split(','):
    crossval = CrossValidation(feature_representations[a_feat_rep_id], text_data_set['label'], 5)

    if options.ann:
      for k_fold in options.folds.split(','):
        k_fold = int(k_fold)
        trn_data, trn_labels, test_data, test_labels = crossval.get_sets(k_fold)

        # ANN classification
        recv_conn, xmit_conn = mp.Pipe()
        # Starting it as a separate process so that, when used with a CUDA-enabled
        # GPU, GPU memory will be reclaimed properly.
        model_proc = mp.Process(
          target=ann_train_and_evaluate,
          args=(trn_data, trn_labels, test_data, test_labels, xmit_conn)
        )
        model_proc.start()
        fold_metrics = recv_conn.recv()
        model_proc.join()
        print("fold {:d} metrics={}".format(k_fold, fold_metrics))
    if options.svm:
      # SVM
      for k_fold in options.folds.split(','):
        k_fold = int(k_fold)
        trn_data, trn_labels, test_data, test_labels = crossval.get_sets(k_fold)

        # one vs one
        svm_clf = svm.SVC(gamma='scale', decision_function_shape='ovo', random_state=333)
        svm_linear = svm.SVC(gamma='scale', decision_function_shape='ovo', random_state=333, kernel='linear')
        #svm_poly = svm.SVC(gamma='scale', decision_function_shape='ovo', random_state=333, kernel='polynomial')

        # one vs rest
        lin_ovr = svm.LinearSVC()

        m1 = get_svm_metrics(svm_clf, trn_data, trn_labels, test_data, test_labels)
        m2 = get_svm_metrics(svm_linear, trn_data, trn_labels, test_data, test_labels)
        #m3 = get_svm_metrics(svm_poly, trn_data, trn_labels, test_data, test_labels)
        m4 = get_svm_metrics(lin_ovr, trn_data, trn_labels, test_data, test_labels)

        print("RBF OVO Accuracy for fold:", k_fold, "is:", m1)
        print("Linear OVO Accuracy for fold:", k_fold, "is:", m2)
        print("Poly OVO Accuracy for fold:", k_fold, "is:", m3)
        print("Linear OVR Accuracy for fold:", k_fold, "is:", m4)
        pass






    if options.bayes:
      # TODO: Add Naive Bayes classification
      pass


class DoAllAction(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    if nargs is not None:
      raise ValueError('nargs not allowed in DoAllAction')
    super(DoAllAction, self).__init__(option_strings, dest, nargs=0, **kwargs)
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, 'ann', True)
    setattr(namespace, 'svm', True)
    setattr(namespace, 'bayes', True)

class DoAllAction(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    if nargs is not None:
      raise ValueError('nargs not allowed in DoAllAction')
    super(DoAllAction, self).__init__(option_strings, dest, nargs=0, **kwargs)
  def __call__(self, parser, namespace, values, option_string=None):
    setattr(namespace, 'ann', True)
    setattr(namespace, 'svm', True)
    setattr(namespace, 'bayes', True)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='UCF NLP Bully Detection System')
  parser.add_argument('input_file', nargs=1, help='Data set input file.')
  parser.add_argument('-n', '--ann', dest='ann', action='store_true', default=False, help='Perform ANN classification.')
  parser.add_argument('-s', '--svm', dest='svm', action='store_true', default=False, help='Perform SVM classification.')
  parser.add_argument('-b', '--bayes', dest='bayes', action='store_true', default=False, help='Perform naive bayes classification.')
  parser.add_argument('-a', '--all', action=DoAllAction, help='Do all classification tasks.')
  parser.add_argument('-f', '--folds', dest='folds', help='Comma-separated list of folds to execute (default: all)', default='0,1,2,3,4')
  parser.add_argument('-g', '--ngrams', dest='ngrams', help='Comma-separated list of n-grams to use (default: all)', default='3-gram,4-gram,3 and 4-gram')

  options = parser.parse_args(sys.argv[1:])
  
  main(options.input_file[0], options)

# vim: set ts=2 sw=2 expandtab:
