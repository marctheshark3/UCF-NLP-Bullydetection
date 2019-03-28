from translate import translate_file
import pandas as pd

file_location = "/Users/marctheshark/Documents/Github/NLP/Tokenization/data.txt"
output = "/Users/marctheshark/Documents/Github/NLP/Tokenization/output.txt"


def n_gram_GetTraining(data, n):

    nothing = preprocessing(data)
    labels, corpus = nothing
    corpus_ngram =[]

    #looking through each sentence in the corpus
    for i in range(len(corpus)):

        sentences = corpus[i]
        sentence_ngram =[]
        #print(sentences)

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

            #if there are unequal ngrams dont use them
            try:
              if len(next_index) < n:
                #print('breaking')
                break
            except:
              w = 0

            if next_index not in tokens:
                tokens.append(next_index)
    tokens.sort()

    encoded_data = np.zeros([len(corpus_ngram), len(tokens)])
    count = 0
    for z in range(len(tokens)):
      unique_token = tokens[z]
      for e in range(len(corpus_ngram)):
        if unique_token in corpus_ngram[e]:
          count += 1
          encoded_data[e, z] = count

    return encoded_data , labels

