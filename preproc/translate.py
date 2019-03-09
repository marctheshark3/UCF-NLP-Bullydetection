# -*- coding: utf-8 -*-

"""Module containing the facilities that translate an entire data set file."""

import pandas as pd

from . import replace_emojis
from . import tokenize_string

def transform_tweet(source_tweet):
  """
  Perform transformation on one tweet, producing a new, transformed tweet.

  :param source_tweet: Tweet text to transform
  :type source_tweet: str
  :return: Transformed tweet text
  :rtype: str
  """
  no_emojis = replace_emojis(source_tweet)
  as_tokens = tokenize_string(no_emojis)
  return ' '.join(as_tokens)

def translate_file(source_file, target_file):
  """
  Translate the content of an input file per the Cyber-bullying Detection
  System pre-processing rules and send said content to a different output file.

  note:: Both input and output files are assumed to be open and ready for
  interaction.

  :param source_file: File-like object containing the original input.
  :param target_file: File-like object that will receive the translated output.
  """
  stream_reader = pd.read_csv(source_file, chunksize=10)
  write_header = True
  for a_chunk in stream_reader:
    transformed_tweets = a_chunk['tweet'].apply(transform_tweet)
    output_chunk = pd.DataFrame(
      {
        'tweet': transformed_tweets,
        'label': a_chunk['label']
      }
    )
    output_chunk.to_csv(target_file, header=write_header, index=False)
    write_header = False

# vim: set ts=2 sw=2 expandtab:
