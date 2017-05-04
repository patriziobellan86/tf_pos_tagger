#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:32:51 2017

@author: patrizio
"""
#
##Import Library
#from sklearn import svm
##Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
## Create SVM classification object 
#model = svm.SVC(kernel='linear', C=1, gamma=1) 
## there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
#model.fit(X, y)
#model.score(X, y)
##Predict Output
#predicted= model.predict(x_test)
#
#sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could
 # avoid this ugly slicing by using a two-dim dataset
y = iris.target
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=11, gamma=10).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()






iris = datasets.load_iris()
X = iris.data#[:, :2] # we only take the first two features. We could
 # avoid this ugly slicing by using a two-dim dataset
y = iris.target
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=11, gamma=10).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel(),xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


#==============================================================================
# new
#==============================================================================
from Word2Vec import Word2Bigrams, Pos2Vec




def nVowels (word):
    vowels='a e i o u'.split()
    nvowels = 0
    
    for i in range(len(word)):
        if word[i] in vowels:
            nvowels += 1 
    return nvowels
def FeaturesExtractor (word):
    f1 = word[-1]   # last character
    f2 = word[-2:] # last 2 char
    f3 = word[-3:]
    f4 = word[0]
    f5 = word[:2]
    f6 = word[:3]
    f7 = nVowels(word) #voc 
    f8 = len(word) # length
    f9 = f8 - f7  #conson
    
    return {'finale':f1,'finale-2':f2,'suffisso':f3,'iniziale':f4,'iniziale-2':f5,'prefisso':f6}


w2v = Word2Bigrams()
p2v = Pos2Vec()


from sklearn.feature_extraction import DictVectorizer
import random


# words for feeding the nn
words = list(p2v.words.keys())

# testing, only the first 5000 words
words = words[:15000]
random.shuffle(words)
print (len(words))
words = [w for w in words if w2v._ValidateWord(w)]
print (len(words))

vec = DictVectorizer(sparse=True)
   
# vettorizzazione dati ingresso
wordsVec = []
for w in words:
    wdict = FeaturesExtractor(w)
    wordsVec.append(wdict)
X = vec.fit_transform(wordsVec)                

Y = []
Ynp = []
poss = []
for w in words:
    pos = p2v.words[w]
    poss.append(pos)
    pos = p2v.Pos2Vec(pos)
    Y.append(pos)
    Ynp.append(np.array(pos))

s=svm.SVC(decision_function_shape='ovr')
y=Y
y=np.array(y)             




C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=11, gamma=10).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()                
                
                
                