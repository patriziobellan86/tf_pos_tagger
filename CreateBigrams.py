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

import numpy as np
import dill

class CreateBigrams (object):
    """
        This class create bigrams
    """
    start_tag='#'
    end_tag='#'
    bidictFilename = 'bigramsDict.pkl'
    validcharsFilename = 'validchars'
    vectorsFilename = 'dict_vectors.pkl'
    
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
            self.__SaveSerializedData(self.vectors, self.vectorsFilename)
        
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
                
        t={}
        for n,w in enumerate(words):
            try:
                print (n,'/',len(words),w)
                w = self.VectorizeWord(w)
                if w:
                    print ('valid', w)
                    t.update(w)
                else:
                    print ('invalid')
            except:
                pass
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
        for bi in self.__Word2Bigrams(word):
            vector[0,self.dictBigrams[bi]] = 1
            
        return {word:vector}

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


if __name__ == '__main__':
    import morphItDataExtractor
    morphit = morphItDataExtractor.MorphItDataExtractor('morphitUtf8.txt') 
    w2=morphit.Words()
    print (len(w2))
    a = CreateBigrams(w2)