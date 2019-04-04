import sys
import os
import multiprocessing as mp
import pandas as pd
from datetime import datetime
from preproc import translate_file
from featrep import encode_tweets
from classify import CrossValidation, ann_train_and_evaluate

def main(input_file_name):
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
  text_data_set = pd.read_csv(preproc_file_name)

  # In this example, we're doing the conversion to feature representations and
  # writing to disk. The logic can be easily converted to continue the other
  # tasks (e.g., classification, statistical significance, etc.)
  feature_representations = {
    '3-gram': encode_tweets(text_data_set['tweet'], [3,]),
    '4-gram': encode_tweets(text_data_set['tweet'], [4,]),
    '3 and 4-gram': encode_tweets(text_data_set['tweet'], [3, 4,]),
  }

  for feat_rep_name, a_feat_rep in feature_representations.items():
    print('{:s} shape: {}'.format(feat_rep_name, a_feat_rep.shape))
  
  crossval = CrossValidation(feature_representations['3-gram'], text_data_set['label'], 5)
  for k_fold in range(5):
    trn_data, trn_labels, test_data, test_labels = crossval.get_sets(k_fold)
    print('{:d} -> trn_data.shape: {}, trn_labels.shape: {}, test_data.shape: {}, test_labels.shape: {}'.format(k_fold, trn_data.shape, trn_labels.shape, test_data.shape, test_labels.shape))

    # ANN classification
    recv_conn, xmit_conn = mp.Pipe()
    model_proc = mp.Process(
      target=ann_train_and_evaluate,
      args=(trn_data, trn_labels, test_data, test_labels, xmit_conn)
    )
    model_proc.start()
    fold_metrics = recv_conn.recv()
    model_proc.join()
    print("fold {:d} metrics={}".format(k_fold, fold_metrics))
    

if __name__ == '__main__':
  # First argument assumed to be the raw data set file name
  main(sys.argv[1])

# vim: set ts=2 sw=2 expandtab:
