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
#iris = datasets.load_iris()
#X = iris.data[:, :2] # we only take the first two features. We could
# # avoid this ugly slicing by using a two-dim dataset
#y = iris.target
## we create an instance of SVM and fit out data. We do not scale our
## data since we want to plot the support vectors
#C = 1.0 # SVM regularization parameter
#svc = svm.SVC(kernel='rbf', C=11, gamma=10).fit(X, y)
## create a mesh to plot in
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#h = (x_max / x_min)/100
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# np.arange(y_min, y_max, h))
#plt.subplot(1, 1, 1)
#Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max())
#plt.title('SVC with linear kernel')
#plt.show()
#
#
#
#
#
#
#iris = datasets.load_iris()
#X = iris.data#[:, :2] # we only take the first two features. We could
#xp=X
#              # avoid this ugly slicing by using a two-dim dataset
#y = iris.target
## we create an instance of SVM and fit out data. We do not scale our
## data since we want to plot the support vectors
#C = 1.0 # SVM regularization parameter
#svc = svm.SVC(kernel='rbf', C=11, gamma=10).fit(X, y)
#xd=X
## create a mesh to plot in
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#h = (x_max / x_min)/100
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
# np.arange(y_min, y_max, h))
#plt.subplot(1, 1, 1)
#Z = svc.predict(np.c_[xx.ravel(), yy.ravel(),xx.ravel(), yy.ravel()])
#Z = Z.reshape(xx.shape)
#plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
#plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
#plt.xlabel('Sepal length')
#plt.ylabel('Sepal width')
#plt.xlim(xx.min(), xx.max())
#plt.title('SVC with linear kernel')
#plt.show()
#
#xp==xd
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
    
    return {'finale':f1,'finale-2':f2,'suffisso':f3,'iniziale':f4,'iniziale-2':f5,'prefisso':f6,'nvocali':f7,'lunghezza':f8,'nconsonanti':f9}


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

#from sklearn.decomposition import PCA

n_components = 2

from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svd = TruncatedSVD(n_components=n_components)
svd.fit(X) 
print(svd.explained_variance_ratio_) 
print(svd.explained_variance_ratio_.sum())
X_svd = svd.fit_transform(X) 
X = X_svd

Y = []
for w in words:
    pos = p2v.words[w]
    pos = p2v.posdict[pos]
    Y.append(pos)
    
s=svm.SVC(decision_function_shape='ovr')
y=Y
y=np.array(y)

xPix = 1240
yPix = 760

xSize = 25 #inches
ySize = xSize/xPix*yPix
             



for c in [1, 10, 100, 1000]:
    for g in [1, 10, 100]:
        svc = svm.SVC(kernel='linear', C=c, gamma=g).fit(X, y)
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = (x_max / x_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        plt.subplot(1, 1, 1)
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        maccuracy = svc.score(X, y) 	#Returns the mean accuracy on the given test data and labels.
        print('SVC Accuracy: ',maccuracy)
        plt.style.use('ggplot')    
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('POS')
        plt.ylabel('Features')
        plt.xlim(xx.min(), xx.max())
        s = 'SVC linear c: '+str(c)+' gamma: '+str(g)
        plt.title(s)
        plt.legend(loc="lower right")
        plt.show()
        plt.gcf().set_inches_size(xSize,ySize)
        plt.gcf().savefig(s,dpi=xSize*6)   # dovrebbe essere dpi=xSize/xPix ma va in errore, così funziona       
        
        
                        
        
        svc = svm.SVC(kernel='rbf', C=c, gamma=g).fit(X, y)
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = (x_max / x_min)/100
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        plt.subplot(1, 1, 1)
        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        maccuracy = svc.score(X, y) 	#Returns the mean accuracy on the given test data and labels.
        print('SVC Accuracy: ',maccuracy)
        plt.style.use('ggplot')    
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('POS')
        plt.ylabel('Features')
        plt.xlim(xx.min(), xx.max())
        s = 'SVC rbf c: '+str(c)+' gamma: '+str(g)
        plt.title(s)
        plt.legend(loc="lower right")
        plt.show()
        plt.gcf().set_inches_size(xSize,ySize) 
        plt.gcf().savefig(s,dpi=xSize*6) # dovrebbe essere dpi=xSize/xPix ma va in errore, così funziona              

        for d in [1, 3, 5]:        
            svc = svm.SVC(kernel='poly', C=c, gamma=g,degree=d).fit(X, y)
            # create a mesh to plot in
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            h = (x_max / x_min)/100
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
            plt.subplot(1, 1, 1)
            Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            maccuracy = svc.score(X, y) 	#Returns the mean accuracy on the given test data and labels.
            print('SVC Accuracy: ',maccuracy)
            plt.style.use('ggplot')    
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.xlabel('POS')
            plt.ylabel('Features')
            plt.xlim(xx.min(), xx.max())
            s = 'SVC poly c: '+str(c)+' gamma: '+str(g) + ' degree: '+str(d)
            plt.title(s)
            plt.legend(loc="lower right")
            plt.show()  
            fig = plt.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.savefig('test2png.png', dpi=100)   
            plt.gcf().set_inches_size(xSize,ySize)
            plt.gcf().savefig(s,dpi=xSize*6) # dovrebbe essere dpi=xSize/xPix ma va in errore, così funziona          

# leggendo una serie di tutorial ho trovato che il classifier più adatto per il nostro task (Multiclass classifier)
# sarebbe un OneVsRestClassifier quindi ho inserito anche questa parte di codice,
# credo che ci siano anche altri tipi di classifier oltre a questo e al SVC che abbiamo usato
# nel for loop precedente, potremmo provarli per vedere qual'è il migliore

plt.style.use('ggplot')        
c=1
g=1
classif = OneVsRestClassifier(SVC(kernel='linear')).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
Z = classif.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
maccuracy = classif.score(X, y) 	#Returns the mean accuracy on the given test data and labels.
print('SVC Accuracy: ',maccuracy)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('POS')
plt.ylabel('Features')
plt.xlim(xx.min(), xx.max())
plt.legend(loc="lower right")
s = 'Multilabel Classifier c: '+str(c)+' gamma: '+str(g)
plt.title(s)
plt.gcf().set_size_inches(xSize,ySize)
plt.gcf().savefig(s,dpi=xSize*6) # dovrebbe essere dpi=xSize/xPix ma va in errore, così funziona       

       
       
# Nei prossimi giorni vorrei provare a fare un multiclass classifier assegnando diversi colori
# alle diverse features come fanno in questi links http://scikit-learn.org/stable/modules/multiclass.html
# http://scikit-learn.org/stable/auto_examples/plot_multilabel.html#sphx-glr-auto-examples-plot-multilabel-py 

#questa funzione definisce l'iperpiano
def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)

# qui creo una lista per ogni classe (POS) nel nostro training set
A=[]
for i in range(len(Y)):
    if Y[i]==1:
     A.append(1)
    else:
     A.append(0)
     
B=[]
for i in range(len(Y)):
    if Y[i]==2:
     B.append(1)
    else:
     B.append(0)
     
C=[]
for i in range(len(Y)):
    if Y[i]==7:
     C.append(1)
    else:
     C.append(0)
     
D=[]
for i in range(len(Y)):
    if Y[i]==10:
     D.append(1)
    else:
     D.append(0)
     
E=[]
for i in range(len(Y)):
    if Y[i]==1:
     E.append(13)
    else:
     E.append(0)
     
F=[]
for i in range(len(Y)):
    if Y[i]==1:
     F.append(20)
    else:
     F.append(0)

# qui creo una tupla con le posizioni dei rispettivi POS     
zero_class = np.where(A)
one_class = np.where(B)
two_class = np.where(C)
three_class = np.where(D)
four_class = np.where(E)
five_class = np.where(F)
#qui faccio un plot per tutti i dati
plt.scatter(X[:, 0], X[:, 1], s=640, c='gray')
#qui faccio un plot per ciascuna delle classi nel training set
plt.scatter(X[zero_class, 0], X[zero_class, 1], s=320, edgecolors='b',
           facecolors='none', linewidths=2, label='Class 1')
plt.scatter(X[one_class, 0], X[one_class, 1], s=240, edgecolors='orange',
           facecolors='none', linewidths=2, label='Class 2')
plt.scatter(X[two_class, 0], X[two_class, 1], s=160, edgecolors='green',
           facecolors='none', linewidths=2, label='Class 3')
plt.scatter(X[three_class, 0], X[three_class, 1], s=80, edgecolors='red',
           facecolors='none', linewidths=2, label='Class 4')
plt.scatter(X[four_class, 0], X[four_class, 1], s=40, edgecolors='purple',
           facecolors='none', linewidths=2, label='Class 5')
plt.scatter(X[five_class, 0], X[five_class, 1], s=20, edgecolors='yellow',
           facecolors='none', linewidths=2, label='Class 6')

min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

min_y = np.min(X[:, 1])
max_y = np.max(X[:, 1])

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X, Y)


plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                'Boundary\nfor class 1')
plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                'Boundary\nfor class 2')

plt.xticks(())
plt.yticks(())
plt.legend()
plt.show()  

#però il grafico esce su una linea non so per quale motivo