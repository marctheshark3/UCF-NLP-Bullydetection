import numpy as np
import pandas as pd
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

file_location = "/Users/marctheshark/Documents/Unity/Github Desktop/NLP/Tokenization/data.txt"

#opening and then closing the file
def getfile(address):
    file = open(address, 'rt')

    data = file.readlines()
    file.close()

    return data

#Looping through the data to pull each sentence and sentiment label
def preprocessing(your_data):
    data = getfile(your_data)
    sentiment = []
    corpus = []
    #looping over the length of the data
    for i in range(len(data)):
        index = data[i]
        # print(index, 'i')

        #reverse looping through each index to find the label 0,1,2,3
        for j in reversed(range(len(index))):
            single_index = index[j]


            try:

                #if the single_index is 0,1,2,3 lets store it
                if int(single_index) <= 3:
                    sentiment.append(int(single_index))
                    #stop this from looping through the rest of the index
                    #since we know we got our label
                    break
            except ValueError:
                w = 0
        #Building sentences from the data and removing punctuation and lowering captilizations.
        words = word_tokenize(index)


        words = [word.lower() for word in words]
        # removing any punctuation
        matrix = str.maketrans('', '', string.punctuation)
        removed = [word.translate(matrix) for word in words]
        filter_words = [word for word in removed if word.isalpha()]

        corpus.append(filter_words)
        #building the unique word bag

        token=[]
        #findthing the unique ngram token for the given corpus
        for words in corpus:
            token.extend(words)
        unique_token = list(set(token))

    return sentiment, corpus, unique_token

print(len(preprocessing((file_location))[0]))
print(preprocessing((file_location))[1][1])
