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

# modules for plotting data
import matplotlib.colors as pltc
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches

import numpy as np               # module for numeric fast computation

import random                    # module for random/shuffling functions

# modules for machine learning
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD
# classes for vectorizing words
from Word2Vec import Word2Bigrams, Pos2Vec

#==============================================================================
# global params
#==============================================================================
n_components = 2

xPix = 1240
yPix = 760

xSize = 25 #inches
ySize = xSize/xPix*yPix
      
trainDim = 0.7
testDim = 0.2  


mapcolors={0:'black', 1:'gray',2:'silver',3:'white',4:'firebrick',
           5:'salmon',6:'darksalmon',7:'red',8:'tomato',9:'gold',
           10:'yellow',11:'sienna',12:'sandybrown',13:'darkorange',
           14:'orange',15:'green',16:'forestgreen',17:'cyan',
           18:'royalblue',19:'blue',20:'navy',21:'magenta',22:'crimson'}

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
#  Function for create and save a figure
#==============================================================================

def CreateFigure(svc, X,Y, s, mapcolors):
    # item colors    
    colors = [mapcolors[c] for c in Y]
    # legend items
    legitems = [mpatches.Patch(color=mapcolors[i], label=p2v.PosFromIndex(i)) for i in range(len(p2v.posdict.keys()))]
       
    # create mesh to plot 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1    # x limits
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1    # y limits
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    plt.figure(figsize=(xSize,ySize)) 
    
    #plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Mean accuracy on the given test data and labels.
    #test_accuracy = svc.score(X_test, y_test) 	
    #print(s,' Accuracy: ',test_accuracy)
    
    # plot style preferences
    plt.style.use('ggplot')   
    plt.contourf(xx, yy, Z)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=pltc.ListedColormap(colors))
    plt.xlabel('Part Of Speech', fontsize=25)
    plt.ylabel('Features', fontsize=25)
    plt.xlim(xx.min(), xx.max())
    plt.title(s, fontsize=32)
    # add legend
    plt.legend(handles=legitems, loc='best', bbox_to_anchor=(1, 1),
               prop={'weight':'roman','size':'xx-large'})
    # Save Figure
    plt.savefig(s+'.jpeg',dpi=xSize*6,format='jpeg', bbox_inches="tight")

#==============================================================================
#  Tesst for Accuracy
#==============================================================================
def TestAccuracy(svc, X_test, y_test):
    test_accuracy = svc.score(X_test, y_test) 	
    print(s,'\n Accuracy: ',test_accuracy)
    return test_accuracy

#==============================================================================
#  script data
#==============================================================================
w2v = Word2Bigrams()
p2v = Pos2Vec()

# words list
words = list(p2v.words.keys())
#
## testing, only the first 5000 words
#words = words[:5000]

# shuffling words in order to avoid any order effect
random.shuffle(words)
# filtring words - only valid words 
words = [w for w in words if w2v._ValidateWord(w)]
print ('total words :',len(words))
# splitting data into training and test
words_training = words[:int(trainDim*len(words))]
words_test = words[int(trainDim*len(words)):int((trainDim+testDim)*len(words))]
print ('training set:', len(words_training))
print ('test set:', len(words_test))

# only for testing phase, use python -c filename to bypass this instrunction
assert (len(words_training)>0 and len(words_test)>0)

# vectorizer
vec = DictVectorizer(sparse=True)
  
#==============================================================================
#  compute dataset for training data 
#==============================================================================

# vectorizing words with features
wordsVec = []
for w in words_training:
    wdict = FeaturesExtractor(w)
    wordsVec.append(wdict)    
X = vec.fit_transform(wordsVec)                
# Single Value Decomposition
svd = TruncatedSVD(n_components=n_components)
# fitting data for Single Value Decomposition
svd.fit(X) 
# Dimensionality reduction on X.
X = svd.fit_transform(X) 
# Y array
Y = []
for w in words_training:
    pos = p2v.words[w]
    pos = p2v.posdict[pos]
    Y.append(pos)
y=np.array(Y)

#==============================================================================
#  compute dataset for test data
#==============================================================================

# vectorizing words with features
wordsVec = []
for w in words_test:
    wdict = FeaturesExtractor(w)
    wordsVec.append(wdict)
X_test = vec.fit_transform(wordsVec)  
# Single Value Decomposition
svd = TruncatedSVD(n_components=n_components)
# fitting data for Single Value Decomposition
svd.fit(X_test) 
# Dimensionality reduction on X.
X_test = svd.fit_transform(X_test) 
# Y array
Y_test = []
for w in words_test:
    pos = p2v.words[w]
    pos = p2v.posdict[pos]
    Y_test.append(pos)
y_test=np.array(Y_test)

##==============================================================================
## testing cycle
##==============================================================================

for c in [1, 10, 100, 1000]:
    for g in [1, 10, 100]:
        s = 'SVC linear c: '+str(c)+' gamma: '+str(g)
        # computing kernel
        svc = svm.SVC(kernel='linear', C=c, gamma=g).fit(X, y)
        TestAccuracy(svc, X_test, y_test)
        CreateFigure(svc, X,Y, s, mapcolors)

        s = 'SVC rbf c: '+str(c)+' gamma: '+str(g)
        svc = svm.SVC(kernel='rbf', C=c, gamma=g).fit(X, y)
        TestAccuracy(svc, X_test, y_test)
        CreateFigure(svc, X,Y, s, mapcolors)
 
        for d in [1, 3, 5]:    
            s = 'SVC poly c: '+str(c)+' gamma: '+str(g) + ' degree: '+str(d)
            svc = svm.SVC(kernel='rbf', C=c, gamma=g).fit(X, y)
            TestAccuracy(svc, X_test, y_test)
            CreateFigure(svc, X,Y, s, mapcolors)
 