#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:02:27 2017

@author: patrizio
"""

import CreateBigrams
import morphItDataExtractor

import numpy as np

print ("Preprocessing Data")

# leggo i dati da morphit
print ('caricamento dati da morphit')
morphit = morphItDataExtractor.MorphItDataExtractor('morphitUtf8.txt')
w2=morphit.Words()
print(len(w2))
print ("creazione bigrams in corso...")
bi = CreateBigrams.CreateBigrams()
w2 = bi.FilterWords(w2)
print ('words filtered:',len(w2))
bi.DictVecBigrams(w2)
print ('vectorization of words')
vectors = bi.VectorizeWords(w2)
print('saving vectors')
bi.SaveVectors(vectors)
print('vectors saved')

#
#outData = np.zeros((1,1, dtype=np.int)
#print 
outData = [morphit.isVerb(w) for w in w2]
print ("out data created")
bi.SaveVectorsOut(outData)
print ("end")