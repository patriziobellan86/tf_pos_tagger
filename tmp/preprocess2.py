# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:25:09 2017

@author: Patrizio Bellan
         CIMeC
         Università degli studi di Trento
"""



print ('preprocess2')
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
outData = np.zeros((1,len(w2)), dtype=np.bool)
outData = [morphit.isVerb(w) for w in w2]
print ("out data created")
bi.SaveVectorsOut(outData)
print ("end")

