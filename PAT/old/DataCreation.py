#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:03:42 2017

@author: francesco
"""

import numpy as np
import dill
import CreateBigrams
import morphItDataExtractor
import random
import pandas as pd

class DataCreation (object):
    """
    This class contains a dictionary composed by word, POS tags, output, usage, score
    """
#    bidictFilename = 'bigramsDict.pkl'
    vectorsFilename = 'vectors.csv'
    
    def __init__(self): # inizializza la classe
        
        self.datafilename = 'my_data.pkl' # .pkl per i dati serializzati
        self.proportiontraining = 0.75
        
        self.data = self.LoadData()
        if not self.data:
            self.data = {}
            self.DataUploading()
            self.SaveData(self.data)     
        self.CreateIndexes()
        
        
    def YieldTraining (self):
        for key in self.ws[:self.limitindex]:
            vectIn = self.data[key]['vector']
            vectOut = self.data[key]['output']
            
            yield (vectIn, vectOut)
    
    def YieldTest (self):
        for key in self.ws[self.limitindex:]:
            vectIn = self.data[key]['vector']
            vectOut = self.data[key]['output']
            
            yield (vectIn, vectOut)
        
        
    def CreateIndexes(self):
        
        self.limitindex = int(len(self.data)*self.proportiontraining)
        self.ws = list(self.data.keys())
        random.shuffle(self.ws)
        
    def DataUploading(self): # carica i dati da morphit e crea i vettori 
        # leggo i dati da morphit
        print ('caricamento dati da morphit')
        morphit = morphItDataExtractor.MorphItDataExtractor('morphitUtf8.txt') # crea un'istanza chiamata morphit dalla classe MorphItDataExtractor
        w2=morphit.Words() # estrae tutte le parole da morphit
        print(len(w2))
        print ("creazione bigrams in corso...")
        bi = CreateBigrams.CreateBigrams() # crea un'istanza chiamata bi della classe CreateBigrams
        w2 = bi.FilterWords(w2) # filtra le parole selezionanto soltanto caratteri alfanumerici
        print ('words filtered:',len(w2))
        bi.DictVecBigrams(w2) # costruisco il dizionario interno di traduzione
        print ('vectorization of words')
        #vectors = bi.VectorizeWords(w2) # vettorizzo le parole costruendo un vettore di vettori
        
        poss = morphit.words.values()
        poss = set(poss)
        poss = list(poss)
        dicttradposs={}
        print(poss)
        for i in range(len(poss)):
            print(i, poss[i])
            dicttradposs[poss[i]]=i
        self.dicttradposs=dicttradposs
        
        print ('sto per vettoriazzare la parola')
        for w in w2:
            print (w)
            vector = bi.VectorizeWord(w)
            
            
            
#==============================================================================
#             
#==============================================================================
            
            pos = morphit.words[w]
            
            
            
            output = self.CreateOutputVector(pos)            
            self.data[w] = {'vector':vector, 'output':output, 'pos':pos}
            
    def CreateOutputVector(self,pos):
        vector = np.zeros((1,len(self.dicttradposs)), dtype=np.int)
        vector[0,self.dicttradposs[pos]] = 1     
        return vector
    
    def SaveData (self, data):
        try:
            with open(self.datafilename, 'wb') as f:
                dill.dump(data, f) # dill salva i dati serializzati in un file 
        except:
            print ("error")
            
    def LoadData (self):
        try:
            with open(self.datafilename, 'rb') as f:
                return dill.load(f)
        except:
            return False    
        #creare altre due funzioni savedicttradposs loaddicttradposs
    def LoadCsvDataFrame (self, filename):
        return pd.DataFrame.from_csv(filename).to_dict()

#d = pd.DataFrame.from_csv("vectors.csv")
#c=d.to_dict()
#c['0'][431]    




    
    
#i=1
#for c in b:
#    i+=1
#    if i <10:
#        pass
#    else:
#        break
#        
        
if __name__ == '__main__':
    a=DataCreation()
    print(len(a.data))  
        