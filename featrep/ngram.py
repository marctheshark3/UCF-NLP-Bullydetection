import numpy as np


def find_n_grams(tweet_list, n_gram_size):
  """
  Iterate over every tweet in `tweet_list`, identifying the unique n-grams (of
  size `n_gram_size`) during the iteration. The unique n-grams are then stored
  in an alphabetically sorted list, and returned to the caller.

  :param tweet_list: list()-like object containing the pre-processed text of
                     every tweet, with one tweet per list element.
  :param int n_gram_size: Size of the unique n-grams to find in the tweets.
  :return: list()-like object containing all unique n-grams found, sorted
           alphabetically.
  """

  unique_ngram = []

  # looking through each sentence in the corpus
  for i in range(len(tweet_list)):

    sentences = tweet_list[i]
    sentence_ngram = []
    # print(sentences)

    # looking at each word of the sentence
    for word in range(len(sentences)):
      # storing the ngram
      ngram = sentences[word:word + n_gram_size]

    # storing all ngrams in their respective sentences

      # if there are unequal ngrams dont use them
      try:
       if len(ngram) < n_gram_size:
        # print('breaking')
        pass

      except:
        if ngram not in sentence_ngram:
          sentence_ngram.append(ngram)

    unique_ngram.append(sentence_ngram)
  unique_ngram.sort()

  return unique_ngram

def isolated_encode(tweet_list, n_gram_list):
  """
  Transform each tweet in `tweet_list` into a feature vector containing the
  presence of every unique n-gram in `n_gram_list`. The resulting matrix will
  be of shape `[len(tweet_list), len(n_gram_list)]` (i.e., a rectangular matrix
  with a row for each tweet, and a column per unique n-gram).

  :param tweet_list: list()-like object containing the pre-processed text of
                     every tweet, with one tweet per list element.
  :param n_gram_list: list()-like object containing a sorted set of unique
                      n-grams.
  :return: NumPy 2-dimensional array containing the isolated tweet feature
           representation.
  """

  encoded_data = np.zeros([len(tweet_list), len(n_gram_list)])
  count = 0
  for z in range(len(n_gram_list)):
    unique_token = n_gram_list[z]
    for e in range(len(tweet_list)):
      if unique_token in tweet_list[e]:
        count += 1
        encoded_data[e, z] = count

  return encoded_data

def encode_tweets(tweet_list, n_gram_size_list):
  """
  Transform the tweets in `tweet_list` into a feature vector representation
  (i.e., encoding) based on the n-grams found in the `tweet_list` tweets. The
  `n_gram_size_list` contains a unique set of n-gram sizes to consider when
  encoding tweets. The function constructs feature representations based on a
  single set of n-grams, or a concatenation of representations based on multiple
  n-gram sets. Passing an `n_gram_size_list` with a single element (e.g.,
  `n_gram_size_list = [4,]`) would create feature representations based on the
  presence of n-grams of size as specified in the single `n_gram_size_list`
  element (e.g., 4-grams, based on the previous example). Passing an
  `n_gram_size_list` with more than one element (e.g.,
  `n_gram_size_list = [3, 4,]`) creates feature representations that are a
  concatenation of representations based on the presence of n-grams for each
  size specified (e.g., 3-gram and 4-gram representations, based on the
  previous example).

  :param tweet_list: list()-like object containing the pre-processed text of
                     every tweet, with one tweet per list element.
  :param n_gram_size_list: Set of n-gram sizes to use when building feature
                           representation.
  :return: NumPy 2-dimensional array containing the tweet feature
           representations or `None` if `n_gram_size_list` is empty.
  :raise TypeError: if `n_gram_size_list` is not iterable.
  """
  result = None
  # Logic can be added here later such that previous encodings can be cached,
  # so that if a caller first asks for 3-gram representation, then 4-gram
  # representation, then 3+4-gram representation, the last call will simply use
  # cached representations created in the earlier calls. For now, the logic is
  # not that smart ...

  # For every n-gram size specified in n_gram_size_list ...
  for an_n_gram_size in n_gram_size_list:
    # Find the unique set of n-grams in the tweet list.
    n_gram_set = find_n_grams(tweet_list, an_n_gram_size)
    # Transform the tweets to feature vectors using the acquired n-gram set.
    temp_encode = isolated_encode(tweet_list, n_gram_set)
    # If the final result is not empty ...
    if result is not None:
      # Append the new set of feature vectors to the right of the already-
      # existing vectors. Row dimension is guaranteed to be identical. Only
      # column dimension changes (grows).
      result = np.hstack((result, temp_encode))
    else:
      # Set the result to the new set of feature vectors
      result = temp_encode
  return result

# vim: set ts=2 sw=2 expandtab:
