import pandas as pd
import numpy as np

## n - number of n in ngram, tweets - list of tweets
## outputs the dictionary of ngrams
def create_dictionary(n, tweets):
	dictionary = []
	for line in tweets:
		for n_gram in fancy_split(n, line):
			if not n_gram in dictionary:
				dictionary.append(n_gram)
	dictionary.sort()
	return dictionary

## n - number of n in ngram, line - current tweet
## returns list of n_grams in a single tweet
#### if n == -1 returns n_gram3 + n_gram4
def fancy_split(n, line):
	if n == -1:
		return fancy_split(3,line) + fancy_split(4,line)
	list_of_ns = []
	temp = ''
	for center in range(0,len(line.split())):
		if center+n <= len(line.split()):
			for window in range(0, n):
				temp+=line.split()[center+window]+' '
			list_of_ns.append(temp[:-1])
			temp = ''
	return list_of_ns

## n - number of n in ngram, filename - preprocessed data filename
## returns the vector representation of each word, and labels
def get_n_grams(n, filename):
	stream_reader = pd.read_csv(filename, dtype={'tweet': str, 'label': np.int32}, nrows=20)
	tweets = stream_reader['tweet'].astype(str).values.tolist()
	labels = stream_reader['label'].astype(np.int32).values.tolist()

	dictionary = create_dictionary(n, tweets)
	print(dictionary)
	vectors = np.zeros((len(tweets),len(dictionary)))
	i = 0
	for line in tweets:
		for word in fancy_split(n, line):
			vectors[i, dictionary.index(word)]+=1
		i+=1
	return vectors, labels

##################################################
vectors, labels = get_n_grams(3, "test_data.csv")
print(vectors[0], labels)



