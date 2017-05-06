#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Università degli studi di Trento (Tn) - Italy
Center for Mind/Brain Sciences CIMeC
Language, Interaction and Computation Laboratory CLIC

@author: Patrizio Bellan
         patrizio.bellan@gmail.com
         patrizio.bellan@studenti.unitn.it

         github.com/patriziobellan86


        Francesco Mantegna
        fmantegna93@gmail.com
        
"""

# modules for natural language processing
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
# modules for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# module for randomize
import random
# module for vectorize
from Word2Vec import Word2Bigrams, Pos2Vec

#==============================================================================
#     function for extract features from a given word
#==============================================================================

def FeaturesExtractor (word):
    def nVowels (word):
        vowels='a e i o u'.split()
        nvowels = 0
        
        for i in range(len(word)):
            if word[i] in vowels:
                nvowels += 1 
        return nvowels

    f1 = word[-1]       # last character
    f2 = word[-2:]      # last two char
    f3 = word[-3:]      # last three letters
    f4 = word[0]        # first letter
    f5 = word[:2]       # first two letters
    f6 = word[:3]       # first three letters
    f7 = nVowels(word)  # vowels 
    f8 = len(word)      # length
    f9 = f8 - f7        # consonants
    
    return {'final-1':f1,'final-2':f2,'final-3':f3,'first-1':f4,'first-2':f5,
            'first-3':f6,'nVowels':f7,'length':f8,'nConsonants':f9}


#==============================================================================
# accuracy function
#==============================================================================
def accuracy(classifier, gold):
    results = classifier.classify_many([fs for (fs, l) in gold])
    correct = [l == r for ((fs, l), r) in zip(gold, results)]
    if correct:
        return sum(correct) / len(correct)
    else:
        return 0
    
#==============================================================================
#     linear regression classifier
#==============================================================================
def My_classifier_linear_regression(trainer, features=FeaturesExtractor, data=None, trainDim=0.8):
    """
        data is a list with a pair (word, tag) for each items
    """
    # Construct a list of classified names, using the names corpus.
    trainDim = int(len(data)*trainDim)
    train = data[:trainDim]
    test = data[trainDim:]

    # Train up a classifier.
    print('Training classifier...')
    classifier = trainer( [(features(n), g) for (n, g) in train] )

    # Run the classifier on the test data.
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(n), g) for (n, g) in test])
    print('Accuracy: %6.4f' % acc)

    # For classifiers that can find probabilities, show the log
    # likelihood and some sample probability distributions.
    try:
        test_featuresets = [features(n) for (n, g) in test]
        pdists = classifier.prob_classify_many(test_featuresets)
        return test, pdists
    except NotImplementedError:
        pass
    return classifier


#==============================================================================
#   script          
#==============================================================================
trainDim = 0.8

w2v = Word2Bigrams()
p2v = Pos2Vec()

words = [w for w in p2v.words.keys() if w2v._ValidateWord(w)]
print ('total words :', len(words))

# restricting set size
#words = words[:150000]

random.shuffle(words)
trainDim = int(trainDim*len(words))
train_set, test_set = words[:trainDim], words[trainDim:]

print('training:',len(train_set))
print('test:', len(test_set))
print ('datasets creation')
train_set = [(FeaturesExtractor(w), p2v.words[w]) for w in train_set]
test_set = [(FeaturesExtractor(w), p2v.words[w]) for w in test_set]

print ('Bayes training')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print ('Bayes classifier')
print('Bayes accuracy', nltk.classify.accuracy(classifier, test_set))

#classifier.classify(gender_features('Neo'))
classifier.show_most_informative_features(150)

# decision tree
print ('decision tree')
classifier = nltk.DecisionTreeClassifier.train(train_set)
print ('decision tree accuracy:',nltk.classify.accuracy(classifier, test_set))
classifier.pseudocode(depth=5)


words = [w for w in p2v.words.keys() if w2v._ValidateWord(w)]
random.shuffle(words)
#words = words[:15000]

data = [(w, p2v.words[w]) for w in words]

for c in [1, 10, 100, 1000, 10000]:
    print ('Logistic Regression', c)
    test_svm, pdists_svm = My_classifier_linear_regression(
            SklearnClassifier(LogisticRegression(C=c)).train, 
            features=FeaturesExtractor, data=data)
        

print ('BernoulliNB binarize = False')
test_bern, pdists_bern = My_classifier_linear_regression(
        SklearnClassifier(BernoulliNB(binarize=False)).train, 
        features=FeaturesExtractor, data=data)


print ('BernoulliNB binarize = True')
test_bern, pdists_bern = My_classifier_linear_regression(
        SklearnClassifier(BernoulliNB(binarize=True)).train, 
        features=FeaturesExtractor, data=data)


print ('MultinomialNB')
test_multiNB, pdists_multiNB = My_classifier_linear_regression(
        SklearnClassifier(MultinomialNB()).train, 
        features=FeaturesExtractor, data=data)
    

