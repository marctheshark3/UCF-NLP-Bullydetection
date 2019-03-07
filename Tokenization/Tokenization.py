import numpy as np
import pandas as pd
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import time

start = time.time()

print("hello")
#change your file location resprective of your path
file_location = "/Users/marctheshark/Documents/NLP/Final Project/data.txt"

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
        sw = set(stopwords.words('english'))
        filter_words = [w for w in filter_words if not w in sw]
        corpus.append(filter_words)

    return sentiment, corpus

def n_gram_GetTraining(data, n):

    nothing = preprocessing(data)
    labels, corpus = nothing
    corpus_ngram =[]

    #looking through each sentence in the corpus
    for i in range(len(corpus)):
        sentences = corpus[i]
        sentence_ngram =[]

        #looking at each word of the sentence
        for word in range(len(sentences)):
            #storing the ngram
            sentence_ngram.append(sentences[word:word+n])
        #storing all ngrams in their respective sentences
        corpus_ngram.append(sentence_ngram)


    #splitting into unique ngram tokens
    tokens = []
    for j in range(len(corpus_ngram)):
        index = corpus_ngram[j]
        for w in range(len(index)):
            next_index = index[w]

            if next_index not in tokens:
                tokens.append(next_index)
    tokens.sort()

    return corpus_ngram , labels, tokens


#data is the location of you file in txt format
#wasnt able to link it from the github not sure how to do that -Marc

tri_gram = n_gram_GetTraining(file_location,3)
quad_gram = n_gram_GetTraining(file_location,4)

combination_gram = np.hstack((tri_gram, quad_gram))

x , y ,z = quad_gram

#might be a good idea to output all three of n_grams so we dont have to reprocess them everytime

end = time.time()
print(end - start)
