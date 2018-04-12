# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 03:32:30 2018

@author: vipin
"""
# Import Libraries
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import twitter_samples as ts
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
import string
from nltk.probability import FreqDist
from itertools import chain
from nltk.stem import SnowballStemmer
print(" ".join(SnowballStemmer.languages))
snowball_stemmer = SnowballStemmer('english')
nltk.download("stopwords")
stopwordsList=set(stopwords.words('english'))
stopwordsListUpdated=list(stopwordsList)
punctuationList=[',','"', '--',"'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '\\', '//', '~', '`', '@', '#', '$', '%', '^', '*', '_', '-', '<', '>', '+', '=','...','***'];
stopwordsListUpdated+= punctuationList

# This is how Naive Bayes Classifier expects the input
#def create_word_features(words):
#    useful_words = [word for word in words if word not in stopwordsListUpdated and not word.isdigit()]
#    my_dict = dict([(word, True) for word in useful_words])
#    return my_dict

#Negative Reviews
#neg_reviews = []
#for fileid in movie_reviews.fileids('neg'):
#    words = movie_reviews.words(fileid)
#    neg_reviews.append((create_word_features(words), "negative"))
    

#print(neg_reviews[0])
#print(len(neg_reviews))

# Positive Reviews
#pos_reviews = []
#for fileid in movie_reviews.fileids('pos'):
#    words = movie_reviews.words(fileid)
#    pos_reviews.append((create_word_features(words), "positive"))
    
#print(pos_reviews[0])
#print(len(pos_reviews))

# Split into Training Set and Testing Set
#train_set = neg_reviews[:750] + pos_reviews[:750]
#test_set = neg_reviews[750:] + pos_reviews[750:]
vocabulary_mv = [([w for w in movie_reviews.words(i) if w.lower() not in stopwordsListUpdated and w.lower() not in string.punctuation], i.split('/')[0]) for i in movie_reviews.fileids()]
word_features = FreqDist(chain(*[i for i,j in vocabulary_mv]))
word_features=list(word_features.keys())
word_features = word_features[:100]
print(len(vocabulary_mv))
numtrain = int(len(vocabulary_mv) * 75 / 100)
train_set_mv = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in vocabulary_mv[:numtrain]]
test_set_mv = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag  in vocabulary_mv[numtrain:]]

print('Train Set Samples: ', len(train_set_mv))
print('Test Set Samples: ', len(test_set_mv))

vocabulary_mv = [([w for w in movie_reviews.words(i) if w.lower() not in stopwordsListUpdated and w.lower() not in string.punctuation], i.split('/')[0]) for i in movie_reviews.fileids()]
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


#Prepare a unigram feature vector based on the presence or absence of words
def get_unigram_features(data,vocab):
    fet_vec_all = []
    for tup in data:
        single_feat_vec = []
        sent = str(tup[0]).lower() #lowercasing the dataset
        for v in vocab:
            if list(sent).__contains__(v):
                single_feat_vec.append(1)
            else:
                single_feat_vec.append(0)
        fet_vec_all.append(single_feat_vec)
    return fet_vec_all

#Add sentiment scores from sentiwordnet, here we take the average sentiment scores of all words
def get_senti_wordnet_features(data):
    fet_vec_all = []
    for tup in data:
        sent = str(tup[0]).lower()
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
def get_lables_ts(data):
    labels = []
    for tup in data:
        if tup.lower()=="neg":
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
    feat_vec_swn =get_senti_wordnet_features(test_set_mv)
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
training_unigram_features = get_unigram_features(train_set_mv,vocabulary_mv) # vocabulary extracted in the beginning
training_swn_features = get_senti_wordnet_features(train_set_mv)

training_features = merge_features(training_unigram_features,training_swn_features)

training_labels = get_lables(train_set_mv)

test_unigram_features = get_unigram_features(test_set_mv,vocabulary_mv)
test_swn_features=get_senti_wordnet_features(test_set_mv)
test_features= merge_features(test_unigram_features,test_swn_features)

test_gold_labels = get_lables(test_set_mv)


#from sklearn.feature_extraction import DictVectorizer as DV
#v = DV( sparse = False )
#training_labels = v.fit_transform(train_set_mv )
#test_gold_labels = v.transform( test_set_mv ) 

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
real_time_test(nb_classifier,vocabulary_mv)
classifier = NaiveBayesClassifier.train(train_set_mv)
accuracy = nltk.classify.util.accuracy(classifier, test_set_mv)
print(accuracy * 100)
# SVM Classifier
#Refer to : http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

training_features, training_labels = make_classification(n_features=4, random_state=0)
clf = LinearSVC(random_state=0)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
svm_classifier = clf.fit(training_features,training_labels)
predictions = svm_classifier.predict(training_features)
print("Precision of linear SVM classifier is:")
precision = calculate_precision(predictions,training_labels)
print("Training data\t" + str(precision))
predictions = svm_classifier.predict(test_features)
#precision = calculate_precision(predictions,test_gold_labels)
#print("Test data\t" + str(precision))
#Real time tesing
#real_time_test(svm_classifier,vocabulary_mv)

##Decision tree algorithm
from sklearn.tree import DecisionTreeClassifier as dtc
clf_gini = dtc(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
dt_classifier=clf_gini.fit(training_features , training_labels)
predictions = dt_classifier.predict(training_features)
print("Precision of DecisionTreeClassifier is")
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(dt_classifier,vocabulary_mv)
##Implementation of logistice regression
from sklearn.linear_model import LinearRegression as lr
lmModel=lr()
lm_classifier=lmModel.fit(training_features , training_labels)
predictions = lm_classifier.predict(test_features)
print("Precision of LinearRegression is")
precision = calculate_precision(predictions,test_gold_labels)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(lm_classifier,vocabulary_mv)



###twitter sentiment analyzer
import re
nltk.download('words')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
ts.fileids()
twitt_str=ts.strings('tweets.20150430-223406.json')
twitt_token=ts.tokenized('tweets.20150430-223406.json')
#print(twitt_str)
print(len(twitt_token))

data_words = nltk.word_tokenize(str(twitt_str))

data_words=[data_words.lower() for data_words in data_words if data_words.isalpha()]
stemmed = [porter.stem(data_word) for data_word in data_words]
print(stemmed)
print(len(stemmed))

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
#print(wordlist)
len(wordlist)

tweet=[]
for t in stemmed:
        if t.lower() not in stopwordsListUpdated and not t.lower().isdigit() and not t.lower().startswith("http") and not t.lower().startswith("@") and not t.lower().startswith("#") and t.lower() not in re.findall(r'[^\w\s,]', t.lower()):
                  if t.lower() in wordlist:
                      lemma=snowball_stemmer.stem(t.lower())
                      tweet.append(lemma)
         
vocabulary_ts = list(set(tweet))
vocabulary_ts.sort()
print('after vocabulary_ts: ',len(vocabulary_ts))
print('after vocabulary_ts: ',len(set(vocabulary_ts)))
print(vocabulary_ts)

import random                
random.shuffle(vocabulary_ts)
cut_point = int(len(vocabulary_ts) * 0.7)
train_set_ts = vocabulary_ts[:cut_point]
test_set_ts = vocabulary_ts[cut_point:]

print('Train Set Samples: ', len(train_set_ts))
print('Test Set Samples: ', len(test_set_ts))


training_unigram_features_ts = get_unigram_features(train_set_ts,vocabulary_ts) # vocabulary extracted in the beginning
training_swn_features_ts = get_senti_wordnet_features(train_set_ts)
training_features_ts = merge_features(training_unigram_features_ts,training_swn_features_ts)
training_labels_ts = get_lables_ts(train_set_ts)

test_unigram_features_ts = get_unigram_features(test_set_ts,vocabulary_ts)
test_swn_features_ts=get_senti_wordnet_features(test_set_ts)
test_features_ts= merge_features(test_unigram_features_ts,test_swn_features_ts)
test_gold_labels_ts = get_lables_ts(test_set_ts)


# Naive Bayes Classifier 
from sklearn.naive_bayes import MultinomialNB

nb_classifier_ts = MultinomialNB().fit(training_features_ts,training_labels_ts) #training process
predictions_ts = nb_classifier_ts.predict(test_features_ts)

print("Precision of NB classifier is")
predictions_ts = nb_classifier_ts.predict(training_features_ts)
precision_ts = calculate_precision(predictions_ts,training_labels_ts)
print("Training data\t" + str(precision_ts))
predictions_ts = nb_classifier_ts.predict(test_features_ts)
precision_ts = calculate_precision(predictions_ts,test_gold_labels_ts)
print("Test data\t" + str(precision_ts))
#Real time tesing
real_time_test(nb_classifier_ts,vocabulary_ts)
# SVM Classifier
#Refer to : http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

clf = LinearSVC(random_state=0)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
svm_classifier_ts = clf.fit(training_features_ts,training_labels_ts)
predictions_ts = svm_classifier_ts.predict(training_features_ts)

print("Precision of linear SVM classifier is:")
precision_ts = calculate_precision(predictions_ts,training_labels_ts)
print("Training data\t" + str(precision_ts))
predictions_ts = svm_classifier_ts.predict(test_features_ts)
precision_ts = calculate_precision(predictions_ts,test_gold_labels_ts)
print("Test data\t" + str(precision_ts))
#Real time tesing
real_time_test(svm_classifier_ts,vocabulary_ts)

##Decision tree algorithm
from sklearn.tree import DecisionTreeClassifier as dtc
clf_gini = dtc(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
dt_classifier=clf_gini.fit(training_features_ts , training_labels_ts)
predictions = dt_classifier.predict(test_features_ts)
print("Precision of DecisionTreeClassifier is")
precision = calculate_precision(predictions,test_gold_labels_ts)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(dt_classifier,vocabulary_ts)

##Implementation of logistice regression
from sklearn.linear_model import LinearRegression as lr
lmModel=lr()
lm_classifier=lmModel.fit(training_features_ts , training_labels_ts)
predictions = lm_classifier.predict(test_features_ts)
print("Precision of LinearRegression is")
precision = calculate_precision(predictions,test_gold_labels_ts)
print("Test data\t" + str(precision))
#Real time tesing
real_time_test(lm_classifier,vocabulary_ts)



#Your program should give accuracy for both the dataset for the mentioned algorithms as below:
#(Datasets)      (Naive Bayes) (SVM) (Decision-tree)  (Logistic-Regression)  
#movie_review        .78          .48 		.94					.60					
#twitter_dataset     .85			   .25      .60			      .40


#classifier = NaiveBayesClassifier.train(train_set)
#accuracy = nltk.classify.util.accuracy(classifier, test_set)
#print(accuracy * 100)

