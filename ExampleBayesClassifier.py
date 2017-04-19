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
    f5 = word[:1]
    f6 = word[:2]
    
    return {'f1':f1,'f2':f2,'f3':f3,'f4':f4,'f5':f5,'f6':f6}

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

