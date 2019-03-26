import numpy as np
import pandas as pd
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import time

start = time.time()


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
    for i in range(1,len(data)):
        index = data[i]

        #reverse looping through each index to find the label 0,1,2,3
        for j in reversed(range(len(index))):
            single_index = index[j]

            try:
                #if the single_index is 0,1,2,3 lets store it
                if any(char.isdigit() for char in single_index):
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
        #somehow the labels and corpus is off, easily fixable
    return sentiment, corpus
x,y = (preprocessing(file_location))

print(len(x))
print(len(y))


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


#data is the location of you file in txt format
#wasnt able to link it from the github not sure how to do that -Marc

tri_gram = n_gram_GetTraining(file_location,3)
quad_gram = n_gram_GetTraining(file_location,4)

combination_gram = np.hstack((tri_gram, quad_gram))

x , y = quad_gram
print(x)
print(len(x))
print(len(y))
qg = pd.DataFrame(quad_gram)
print(qg)
qg.to_csv(path_or_buf= "/Users/marctheshark/Documents/NLP/Final Project/output.csv" , index=False)

#export_csv = quad_gram.to_csv (r"/Users/marctheshark/Documents/NLP/Final Project/outputdata.csv", header=True) #Don't forget to add '.csv' at the end of the path

#might be a good idea to output all three of n_grams so we dont have to reprocess them everytime

end = time.time()
print(end - start)
'''  
from sklearn.model_selection import train_test_split , GridSearchCV , RepeatedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
print(len(x))
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=333)
print(len(X_train))
print(len(y_train))

kf = KFold(n_splits=10)

hyperparameters = [{'kernel': ['rbf'], 'gamma' : [1, .1, .01, 0.001],  'C': [1, 10, 100]},
                   {'kernel': ['linear'], 'gamma' : [1, .1, .01, 0.001] ,  'C': [1, 10, 100]}]

classifier = GridSearchCV( estimator= svm.SVC(random_state= 333) , param_grid= hyperparameters, cv = kf)
classifier.fit(X_train, y_train)

means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
    print("%0.4f (+/-%0.03f) for %r" % (mean, std * 2, params))
print ('')
print ("The Best Training Score was:" , classifier.best_score_)
print('')
print ("The Best Parameters were: " , classifier.best_params_)
print (' ')
'''