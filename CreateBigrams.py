# -*- coding: utf-8 -*-
"""
                               CreateBigrams.py

This module contains the class to extract and elaborate data from morphit

Everything you need is not here!

Vesion: 1.0-c stable
CODE TESTED ON Python 3.5 and Python 2.7
#==============================================================================
Università degli studi di Trento (Tn) - Italy
Center for Mind/Brain Sciences CIMeC
Language, Interaction and Computation Laboratory CLIC

@author: Patrizio Bellan
         patrizio.bellan@gmail.com
         patrizio.bellan@studenti.unitn.it

         github.com/patriziobellan86


        Francesco Mantegna
        ADD_EMAIL_ADDRESS
        
#==============================================================================
@author: Patrizio Bellan    patrizio.bellan@gmail.com
         Francesco Mantegna ADD_EMAIL_ADDRESS
         
Done:
        change some methods in order to minimize code and make it more flexible
ToDo:
        testing changes
        
Last change: 17/04/2017 Patrizio Bellan        
"""
import pandas as pd
import numpy as np
import dill
import random

class CreateBigrams (object):
    """
        This class create bigrams
    """
    start_tag='#'
    end_tag='#'
    bidictFilename = 'bigramsDict.pkl'
    validcharsFilename = 'validchars'
    vectorsFilename = 'vectors.csv'
    
    def __init__(self, words=None, bidictFilename=None,
                 validcharsFilename=None, vectorsFilename=None):
        self.bidictFilename = bidictFilename or self.bidictFilename 
        self.validcharsFilename = validcharsFilename or self.validcharsFilename
        self.vectorsFilename = vectorsFilename or self.vectorsFilename
        
        # load characters for words validation
        self.validchars = [l.strip() for l in self.__LoadData(self.validcharsFilename)]
        print (self.validchars)
        # try to load dictionary for valid word
        self.dictBigrams = self.__LoadSerializedData(self.bidictFilename)
#        print (len(self.dictBigrams))
        # if the dictionary does not exist, it is impossible to continue untill
        # we calculete it
        if self.dictBigrams:
            self.len_dict_bigrams = len(self.dictBigrams)
            
        self.vectors = self.__LoadSerializedData(self.vectorsFilename)
        if not self.vectors and words:
            self.vectors = self.VectorizeWords(words)
            self.SaveCsvData(self.vectors)
#            self.__SaveSerializedData(self.vectors, self.vectorsFilename)
        
    def ValidateWord (self, word):
        if [False for l in word if l not in self.validchars]:
            return False
        return True
    
    def __Word2Bigrams (self, word):
        bigrams=[]
        # first char
        if not self.ValidateWord(word):
            return False
        try:
            bigrams.append(str(self.start_tag+word[0]))
            # main loop
            for i in range(len(word)-1):
                bigrams.append (str(word[i]+word[i+1]))
            # last char
            bigrams.append(str(word[-1]+self.end_tag))
            
            return bigrams
        except:
            print ('invalid word: ', word, word[i]+word[i+1])
            return False

    def __Words2Bigrams (self, words):
        bigrams=[]
        for word in words:
            word = word.lower()
            vec = self.__Word2Bigrams(word)
            if vec:
                bigrams.extend(vec)
        return bigrams

    def ComputeDictBigrams (self, words):
#        y=self.__Words2Bigrams(words)
#        print('y',len(y))
        self.dictBigrams = self.__VectorizeDictBigrams(
                    self.__Words2Bigrams(words))
        
        self.len_dict_bigrams = len(self.dictBigrams)
        
        return self.__SaveSerializedData(self.dictBigrams, self.bidictFilename)
        
    def __VectorizeDictBigrams ( self, vectors):
#        vec=set(vectors)
#        vec=list(vec)
        return {bi:n for n,bi in enumerate(list(set(vectors)))} #vec)}

    def VectorizeWords(self, words):
        """
            IMPORTANT 
            
            Now this method:
                return a dictionary with word as key and vector as value
        """
        # if the dict is empty, I go and populate it
        if not self.dictBigrams:
            self.ComputeDictBigrams(words)
                
#        t={}
        t=[]
        for n,w in enumerate(words):
#            try:
            print (n,'/',len(words),w)
            w = self.VectorizeWord(w)
            if w:
                print ('valid')
#                    t.update(w)
                t.append(w)
            else:
                print ('invalid')
#            except:
#                pass
        return t
        
    def VectorizeWord(self, word):
        """
            IMPORTANT 
            
            Now this method:
                return a dictionary with word as key and vector as value
        """
        if not self.ValidateWord(word):
            return False
        vector = np.zeros((1,self.len_dict_bigrams), dtype=np.int)
        try:
            for bi in self.__Word2Bigrams(word):
                vector[0,self.dictBigrams[bi]] = 1
                
#            return [{'word':word, 'vector':vector.tolist()}]
            return [word, vector.tolist()[0]]
        except:
            return False
        
#==============================================================================
# new
#==============================================================================

    def __SaveSerializedData (self, data, filename):
        """
           this method save serialized data into file
           
           to do that, it uses dill library to serialized data
           
           input 
               data     data to store
               filename name of the file
            return
                True / save ok
                False/ error during save 
        """
        try:
            with open(filename, 'wb') as f:
                dill.dump(data, f)
            return True
        except:
            print (" error occured during saving on %s"% filename)
            return False
        
    def __LoadSerializedData (self, filename):
        try:
            with open(filename, 'rb') as f:
                return dill.load(f)
        except:
            return False
        
    def __LoadData (self, filename):
        try:
            with open (filename, 'r') as f:
                return f.readlines()
        except:
            return False
        
    def __SaveData (self, data, filename):
        try:
            with open(filename, 'a') as f:
                f.writelines(data)
        except:
            return False

    def SaveCsvData (self, vectors):
        pd.DataFrame(vectors).to_csv(self.vectorsFilename)


class DataCreation (object):
    vectorsFilename = 'vectors.csv'
    def __init__(self, dataoutput_dict, training='training.csv',
                 test='test.csv'): 
        self.dataoutput_dict = dataoutput_dict
        self.outputdict = self.VectorizeDict(dataoutput_dict)
        self.proportiontraining = 0.90
        self.data = self.__LoadCsvDataFrame(self.vectorsFilename)
        self.limitindex = int(len(self.data['0'])*self.proportiontraining)
                
#        self.indexes = self.data['0'].keys()
        self.indexes = list(self.data['0'].keys())
#        self.l=self.l[0]
         
        random.shuffle(self.indexes)
        
    def Save(self):
        pd.DataFrame(self.CreateSet(self.data, 
                            self.indexes[:self.limitindex])).to_csv(training)
        pd.DataFrame(self.CreateSet(self.data, 
                            self.indexes[self.limitindex:])).to_csv(test)
        
        
    def VectorizeDict( self, data_dict):
        labels = list(data_dict.values())
        return {bi:n for n,bi in enumerate(list(set(labels)))}
    
    def CreateOutputVector(self,pos):
        vector = np.zeros((1,len(self.outputdict)), dtype=np.int)
        vector[0,self.outputdict[pos]] = 1     
        return vector
    
    def __LoadCsvDataFrame (self, filename):
        return pd.DataFrame.from_csv(filename).to_dict()

#    def CreateIndexes(self, indexes):
##        self.limitindex = int(len(self.data)*self.proportiontraining)
#        random.shuffle(indexes)
#        return indexes
    
    def CreateSet(self, data, indexes):
        dataset = []
        for i in indexes:
            word = data['0'][i]
            print (i,word)
            
            vector = data['1'][i]
            pos = self.dataoutput_dict[word]
            pos_vect = self.CreateOutputVector(pos)
            
            dataset.append([i, word, vector,pos,pos_vect])
        return dataset
    
        
if __name__ == '__main__':
    import morphItDataExtractor
    morphit = morphItDataExtractor.MorphItDataExtractor('morphitUtf8.txt') 
    w2=list(morphit.Words())
#    w2=w2[:1000]
    w2=[w.lower() for w in w2]
    print (len(w2))
    a = CreateBigrams(w2)
    words_dict={k.lower():v for k,v in morphit.words.iteritems()}
    
    d = DataCreation(words_dict)
    print (d.outputdict)
    print (d.indexes)
#d = pd.DataFrame.from_csv("vectors.csv")
#c=d.to_dict()
#c['0'][431]    
