#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:26:25 2017

@author: patrizio
"""

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import csv
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn


import pandas as pd

# TODO

# load data from dataset and create wrapper functions for batching

# input and output dimensions are read from dataset

training = 'training.csv'
test = 'test.csv'
configuration = 'configuration.ini'


# read configuration file
with open (configuration, 'r') as f:
    config = [int(w) for w in [x.strip() for x in f.readlines()]]
inputnode = config[1]
outputnode = config[-1]

def ReadInput (dataline):
    return [dataline[i] for i in range(1,config[1])]

def ReadOutput (dataline):
    return [dataline[i] for i in range ((config[1]+1),len(dataline))] # read out only the last cell that represent the output vector

def LoadCsvDataFrame (filename):
        vectors = {}
        infile  = open(filename, "r")
        reader = csv.reader(infile, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            vectors[row[0]] = row[1:]
            vectors[row[0]] = {'inputVector': ReadInput(row), 'outputVector':ReadOutput(row), 'outLabel': row[config[1]]}
            
        return vectors    
a=LoadCsvDataFrame(test)


# ATTENTION!!1 LOST ONE CELL AND I DO NOT UNDERSTAND WHERE! INPUT VECTOR LEN < DI 1

training = LoadCsvDataFrame(training)
test = LoadCsvDataFrame(test)

#training.update(test)
n = int(len(training)*0.95)

def FeaturesExtractor (word):
    f1 = word[-1]   # last character
    f2 = word[-2:] # last 2 char
    f3 = word[-3:]
    f4 = word[0]
    f5 = word[:2]
    f6 = word[:3]
    
    
    return {'finale':f1,'finale-2':f2,'suffisso':f3,'iniziale':f4,'iniziale-2':f5,'prefisso':f6}

w='mangiavo'
print (FeaturesExtractor(w))
    



#http://www.nltk.org/book/ch06.html


import nltk
featuresets = [(FeaturesExtractor(w),training[w]['outLabel'] ) for w in training.keys()]
train_set, test_set = featuresets[n:], featuresets[:n]
classifier = nltk.NaiveBayesClassifier.train(train_set)






#>>> classifier.classify(FeaturesExtractor('mangiavo'))
#'VER'
#>>> classifier.classify(FeaturesExtractor('mangiavamo'))
#'VER'
#>>> classifier.classify(FeaturesExtractor('mangiai'))
#'ADJ'
#>>> classifier.classify(FeaturesExtractor('mangiai'))
train_set1 =train_set

def gender_features2(name):
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

featuresets = [(gender_features2(w),training[w]['outLabel'] ) for w in training.keys()]
train_set, test_set = featuresets[n:], featuresets[:n]
classifier2 = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(FeaturesExtractor('mangiavo')))
print(classifier2.classify(gender_features2('mangiavo')))



classifier3 = nltk.classify.DecisionTreeClassifier.train(train_set1, entropy_cutoff=0,support_cutoff=0)
print(classifier3.classify(FeaturesExtractor('mangiavo')))

classifier4 = nltk.classify.DecisionTreeClassifier.train(train_set, entropy_cutoff=0,support_cutoff=0)
print(classifier4.classify(gender_features2('mangiavo')))


from nltk.classify import maxent
featuresets = [(FeaturesExtractor(w),training[w]['outLabel'] ) for w in training.keys()]
train, test = featuresets[n:], featuresets[:n]


encoding = maxent.TypedMaxentFeatureEncoding.train(train, count_cutoff=3, alwayson_features=True)

classifier = maxent.MaxentClassifier.train(train, bernoulli=False, encoding=encoding, trace=0)

featuresets = [FeaturesExtractor(w) for w in training.keys()]
train, test = featuresets[n:], featuresets[:n]

print(classifier.classify_many(test))

postags=[training[w]['outLabel'] for w in training.keys()]
ftable = pd.DataFrame(featuresets)
ftable['postags']=postags
#ftable.columns = sorted(FeaturesExtractor(w).keys(),['postags'])
ftable.head()
print(ftable.head)

from sklearn.feature_extraction import DictVectorizer
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import MultiTaskLassoCV

vec = DictVectorizer()
pos_vectorized = vec.fit_transform(featuresets)
X =  pos_vectorized.toarray()
print(X)
lpostags = [({'postags':training[w]['outLabel']}) for w in training.keys()]
tpos_vectorized =  vec.fit_transform(lpostags)
Y =  tpos_vectorized.toarray()
print(Y)
feature_names = vec.get_feature_names()
print(feature_names)
y=Y[:,6]
#X.shape
#X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)
#X_new.shape
#print(X_new.shape)

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = MultiTaskLassoCV()
# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, Y) # takes a long time
n_features = sfm.transform(X).shape[1]
# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
#while n_features > 2:
#    sfm.threshold += 0.1
X_transform = sfm.transform(X)
#    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from MorphIt using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()

from sklearn.decomposition import PCA

n_components = 2

pca = PCA(n_components=n_components)
pca.fit(X)
X_pca = pca.fit_transform(X)
n_samples, n_features = X_pca.shape

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=n_components)
svd.fit(X) 
print(svd.explained_variance_ratio_) 
print(svd.explained_variance_ratio_.sum())
X_svd = svd.fit_transform(X) 

plt.figure(figsize=(9, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
#plt.plot(X_pca[0:int(len(X_pca)/2),0],X_pca[0:int(len(X_pca)/2),1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
#plt.plot(X_pca[(int(len(X_pca)/2)+1):len(X_pca)-1,0], X_pca[(int(len(X_pca)/2)+1):len(X_pca)-1,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
# Percentage of variance explained for each components

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure(figsize=(9, 5))
plt.scatter(X_svd[:, 0], X_svd[:, 1])

plt.figure()
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MorphIt dataset')


# if one extracts such a context around each individual word of a corpus of documents
# the resulting matrix will be very wide (many one-hot-features) with most of them being
# valued to zero most of the time

#from sklearn.feature_extraction import DictVectorizer
#v = DictVectorizer(sparse=False)
#X = v.fit_transform(featuresets)
#print(X)
#dpostags = [({'postags':training[w]['outLabel']}) for w in training.keys()]
#Y = v.fit_transform(dpostags)
#print(Y)
