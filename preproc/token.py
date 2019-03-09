# -*- coding: utf-8 -*-

"""Module containing tools useful for tokenizing text for pre-processing."""

import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def tokenize_string(orig_string):
  """
  Create a list of string tokens from the input string. The resulting tokens
  exhibit the following traits:

  1. Case normalized (to lower case)
  2. Without punctuation
  3. All alphabetic
  
  In addition, all English stop words, per NLTK, are removed. Finally, words
  are "stemmed" to the common root or base (using the Porter Stemming
  algorithm).

  :param orig_string: String to extract tokens from.
  :type orig_string: str
  :return: List of tokens extracted per the description.
  :rtype: list
  """
  result = word_tokenize(orig_string.lower())
  nopunct_trans = str.maketrans('', '', string.punctuation)
  result = [word.translate(nopunct_trans) for word in result]
  result = [word for word in result if word.isalpha()]
  eng_stop_words = set(stopwords.words('english'))
  result = [word for word in result if not word in eng_stop_words]
  porter_stem = PorterStemmer()
  result = [porter_stem.stem(word) for word in result]

  return result

# vim: set ts=2 sw=2 expandtab:
