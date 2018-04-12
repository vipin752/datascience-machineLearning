# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:47:32 2018

@author: vipin
"""
#import sentiwordnet and wordnet
import string
import os
import nltk
from collections import defaultdict
from itertools import chain
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
import csv
from nltk.probability import FreqDist
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews as mr
mr.fileids()
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples as ts
ts.fileids()
from nltk.corpus import stopwords
nltk.download("stopwords")

ts.strings('tweets.20150430-223406.json')
ts.tokenized('tweets.20150430-223406.json')
#**SnowballStemmer**
#For Snowball Stemmer, which is based on Snowball Stemming Algorithm, can be used in NLTK like this:
from nltk.stem import SnowballStemmer
print(" ".join(SnowballStemmer.languages))
snowball_stemmer = SnowballStemmer('english')

#read the data
pathTrain=os.getcwd();
pathTrain=pathTrain+"\data\sentiment_analysis\Training.csv"
training = os.path.join(pathTrain)
print(training)
#test
pathTest=os.getcwd();
pathTest=pathTest+"\data\sentiment_analysis\Test.csv"
test = os.path.join(pathTest)
print(test)
try:
    reader_train = csv.reader(open(training,'r'))
    reader_test = csv.reader(open(test,'r'))
except IOError:
    reader_train = csv.reader(open("D:\\DataScience\\AEGIS\\NLP\data\\sentiment_analysis\\training.csv",'r'))
    reader_test = csv.reader(open("D:\\DataScience\\AEGIS\\NLP\\data\\sentiment_analysis\\test.csv",'r'))

training_data = []
test_data = []
##add traain data
header = 1
for row in reader_train:
        if header==1:
                header=0
                continue
        training_data.append(row)
#add test data
header=1
for row in reader_test:
        if header==1:
                header=0
                continue
        test_data.append(row)
 # Examples from training data
print(training_data[1])
print(len(training_data), len(test_data))

#Required for Bag of words (unigram features) creation
vocabulary = [x.lower() for tagged_sent in training_data for x in tagged_sent[0].split()]
stopwordsList=set(stopwords.words('english'))
stopwordsListUpdated=list(stopwordsList)
punctuationList=['.', ',',' ','', '"', '--',"'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\\', '//', '~', '`', '@', '#', '$', '%', '^', '*', '_', '-', '<', '>', '+', '=','...','***'];
stopwordsListUpdated+= punctuationList
type(vocabulary)

vocabulary_mv = [([w for w in mr.words(i) if w.lower() not in stopwordsListUpdated and w.lower() not in string.punctuation], i.split('/')[0]) for i in mr.fileids()]
word_features = FreqDist(chain(*[i for i,j in vocabulary_mv]))
word_features=list(word_features.keys())
word_features = word_features[:100]
print(len(vocabulary_mv))
numtrain = int(len(vocabulary_mv) * 75 / 100)
train_set_mv = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in vocabulary_mv[:numtrain]]
test_set_mv = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag  in vocabulary_mv[numtrain:]]

print('before train_set: ',len(train_set_mv))
print('before test_set: ',len(test_set_mv))
#vocabulary
print('before: ',len(vocabulary))
print('before: ',len(set(vocabulary)))
print('before vocabulary_mv: ',len(vocabulary_mv))

vocab=[]
for token in vocabulary:
    vocab.append(word_tokenize(token))
#print(vocab)
len(vocab)

token_updated=[]
for token in vocab:
    if token not in stopwordsListUpdated:
        for tok in token:
            if tok not in stopwordsListUpdated and not tok.isdigit():
                lemma=snowball_stemmer.stem(tok)
                token_updated.append(lemma)
#print(len(vocabulary))
#sorting the list
vocabulary = list(set(token_updated))
vocabulary.sort()
print('after: ',len(vocabulary))
print('after: ',len(set(vocabulary)))
print(vocabulary)
#print(len(vocabulary))
#sorting the list
vocab_mv=[]
for token in vocabulary_mv:
    for tok in token:
        for t in tok:
            vocab_mv.append(word_tokenize(t))
len(vocab_mv)    
token_updated_mv=[]
try:
    for token in vocab_mv:
        if token[0].lower() not in stopwordsListUpdated and not token[0].lower().isdigit() :
            lemma=snowball_stemmer.stem(token[0].lower())
            token_updated_mv.append(lemma)
except:
    print('token n excp: ',token[0])
        
vocabulary_mv = list(set(token_updated_mv))
vocabulary_mv.sort()
print('after vocabulary_mv: ',len(vocabulary_mv))
print('after vocabulary_mv: ',len(set(vocabulary_mv)))
################## Extracting Features #########################
#Prepare a unigram feature vector based on the presence or absence of words
def get_unigram_features(data,vocab):
    fet_vec_all = []
    for tup in data:
        single_feat_vec = []
        sent = tup[0].lower() #lowercasing the dataset
        for v in vocab:
            if ''.join(sent).__contains__(v):
                single_feat_vec.append(1)
            else:
                single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)
    return fet_vec_all

#for movies
def get_unigram_features_mv(data,vocab):
    fet_vec_all = []
    for tup, val in data:
        single_feat_vec = []
        print("\n",tup)
        #lowercasing the dataset
        if not tup[0].isdigit():
            sent = tup[0].str.lower() 
            for v in vocab:
              if sent.__contains__(v):
                  single_feat_vec.append(1)
              else:
                  single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)
    return fet_vec_all

#Add sentiment scores from sentiwordnet, here we take the average sentiment scores of all words
def get_senti_wordnet_features(data):
    fet_vec_all = []
    for tup in data:
        sent = tup[0].lower()
        words = sent.split()
        pos_score = 0
        neg_score = 0
        for w in words:
            senti_synsets = swn.senti_synsets(w.lower())
            for senti_synset in senti_synsets:
                p = senti_synset.pos_score()
                n = senti_synset.neg_score()
                pos_score+=p
                neg_score+=n
                break #take only the first synset (Most frequent sense)
        fet_vec_all.append([float(pos_score),float(neg_score)])
    return fet_vec_all
##Merge the two scores
def merge_features(featureList1,featureList2):
    # For merging two features
    if featureList1==[]:
        return featureList2
    merged = []
    for i in range(len(featureList1)):
        m = featureList1[i]+featureList2[i]
        merged.append(m)
    return merged
#extract the sentiment labels by making positive reviews as class 1 and negative reviews as class 2
def get_lables(data):
    labels = []
    for tup in data:
        if tup[1].lower()=="neg":
            labels.append(-1)
        else:
            labels.append(1)
    return labels
def calculate_precision(prediction, actual):
    prediction = list(prediction)
    correct_labels = [prediction[i]  for i in range(len(prediction)) if actual[i] == prediction[i]]
    precision = float(len(correct_labels))/float(len(prediction))
    return precision
#real time testing method below
def real_time_test(classifier,vocab):
    print("Enter a sentence: ")
    inp = input()
    print(inp)
    feat_vec_uni = get_unigram_features(inp,vocab)
    feat_vec_swn =get_senti_wordnet_features(test_data)
    feat_vec = merge_features(feat_vec_uni, feat_vec_swn)

    predict = classifier.predict(feat_vec)
    if predict[0]==1:
        print("The sentiment expressed is: positive")
    else:
        print("The sentiment expressed is: negative")

################# Training and Evaluation #######################
#Preparing training and test tuples
#The feature_vecor set looks like [featurevector1, featurevector2,...,featurevectorN] where each featurevectorX is a list
#The label set looks like [label1,label2,...,labelN]
training_unigram_features = get_unigram_features(training_data,vocabulary) # vocabulary extracted in the beginning
training_swn_features = get_senti_wordnet_features(training_data)

training_features = merge_features(training_unigram_features,training_swn_features)

training_labels = get_lables(training_data)

test_unigram_features = get_unigram_features(test_data,vocabulary)
test_swn_features=get_senti_wordnet_features(test_data)
test_features= merge_features(test_unigram_features,test_swn_features)

test_gold_labels = get_lables(test_data)

### for movies review data methods
# vocabulary extracted in the beginning
training_unigram_features_mv = get_unigram_features_mv(train_set_mv,vocabulary_mv) 
training_swn_features_mv = get_senti_wordnet_features(train_set_mv)

training_features_mv = merge_features(training_unigram_features_mv,training_swn_features_mv)

training_labels_mv = get_lables(train_set_mv)

test_unigram_features_mv = get_unigram_features(test_set_mv,vocabulary_mv)
test_swn_features_mv=get_senti_wordnet_features(test_set_mv)
test_features_mv= merge_features(test_unigram_features_mv,test_swn_features_mv)

test_gold_labels_mv = get_lables(test_set_mv)

# Naive Bayes Classifier 
from sklearn.naive_bayes import MultinomialNB

nb_classifier = MultinomialNB().fit(training_features,training_labels) #training process
predictions = nb_classifier.predict(test_features)

print("Precision of NB classifier is")
predictions = nb_classifier.predict(training_features)
precision = calculate_precision(predictions,training_labels)
print("Training data\t" + str(precision))
predictions = nb_classifier.predict(test_features)
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(nb_classifier,vocabulary)
# SVM Classifier
#Refer to : http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC(penalty='l2', C=0.01).fit(training_features,training_labels)
predictions = svm_classifier.predict(training_features)

print("Precision of linear SVM classifier is:")
precision = calculate_precision(predictions,training_labels)
print("Training data\t" + str(precision))
predictions = svm_classifier.predict(test_features)
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(svm_classifier,vocabulary)

##Decision tree algorithm
from sklearn.tree import DecisionTreeClassifier as dtc
clf_gini = dtc(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
dt_classifier=clf_gini.fit(training_features , training_labels)
predictions = dt_classifier.predict(test_features)
print("Precision of DecisionTreeClassifier is")
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(dt_classifier,vocabulary)

##Implementation of logistice regression
from sklearn.linear_model import LinearRegression as lr
lmModel=lr()
lm_classifier=lmModel.fit(training_features , training_labels)
predictions = lm_classifier.predict(test_features)
print("Precision of LinearRegression is")
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(lm_classifier,vocabulary)
#Your program should give accuracy for both the dataset for the mentioned algorithms as below:
#(Datasets)      (Naive Bayes) (SVM) (Decision-tree)  (Logistic-Regression)  
#movie_review        ?            ? 		?					?					
#twitter_dataset     ?			 ?		?					?			



