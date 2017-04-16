# -*- coding: utf-8 -*-
"""
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
    validcharsFilename = 'validchars.txt'
    
    def __init__(self, words = None):
        # load characters for words validation
        self.validChars = self.__LoadData(self.validcharsFilename)

        # try to load dictionary for valid word
        self.dictBigrams = self.__LoadSerializedData(self.bidictFilename)
        self.len_dict_bigrams = len(self.dictBigrams)
        
        # if the dictionary does not exist, it is impossible to continue untill
        # we calculete it
        if not self.dictBigrams and words:
            # automatically start dictbigrams computation
            self.ComputeDictBigrams(words)
#        
#        self._SaveWordsDictionary()
#        self.dictBigrams = self.LoadDictBigrams()
        
    def ValidateWord (self, word):
        if [False for l in word if l in self.validchars]:
            return True
        return False
#            
#    def FilterWords (self, words):
#        return [w for w in words if w.isalpha()]

    def Word2Bigrams (self, word):
        bigrams=[]
        # first char
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

    def Words2Bigrams (self, words):
        bigrams=[]
        for word in words:
            # only valid vectorized word
            if self.ValidateWord(word):
                vec = self.Word2Bigrams(word)
                if vec:
                    bigrams.extend(vec)
        return bigrams
#    
#    def __UniqBigrams(self, bigrams):
#        return set(bigrams)    
#    def UniqBigrams (self, words):
#        bigrams = self.Words2Bigrams(words)
#
#        uniq = set()
#        for bis in bigrams:
#            uniq.add(bis)
#        return list(uniq)
    def ComputeDictBigrams (self, words):
        self.dictBigrams = self.VectorizeDictBigrams(
                    self.__Words2Bigrams(words))
        
        self.len_dict_bigrams = len(self.dictBigrams)
        
        return self.__SaveSerializedData(self.dictBigrams, self.bidictFilename)
        
    def __VectorizeDictBigrams ( self, vectors):
        return {bi:n for n,bi in enumerate([set(vectors)])}

        
#        
#    def DictVec (self, words):
#        """
#            dictBigrams lo uso per passare un eventuale stesso dict e non calcolarlo
#        """
#        if not self.dictBigrams:
#            print ('creazione di dict bigrmas')
#            dictBigrams = {bi:n for n,bi in enumerate(self.UniqBigrams(words))}
#            self.dictBigrams = dictBigrams
#            self.SaveDictBigrams()
#            self.len_dict_bigrams = len(self.dictBigrams)
#        return self.dictBigrams

    def VectorizeWords(self, words):
        """
            IMPORTANT 
            
            Now this method:
                return a dictionary with word as key and vector as value
        """
        # if the dict is empty, I go and populate it
        if not self.dictBigrams:
            self.ComputeDictBigrams(words)
            
#        return [self.VectorizeWord(w) for w in words]
        t=[]
        for n,w in enumerate(words):
            print (n,'/',len(words),w)
#            t.append(self.VectorizeWord(w))
            t.update(self.VectorizeWord(w))
        return t
        
    def VectorizeWord(self, word):
        """
            IMPORTANT 
            
            Now this method:
                return a dictionary with word as key and vector as value
        """
        vector = np.zeros((1,self.len_dict_bigrams), dtype=np.int)
        for bi in self.Word2Bigrams(word):
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

#    def _SaveWordsDictionary (self):
#        """
#            This method save the dictionary of bigrams for valid words
#            
#            this step is a must in order to fix input layer's dimension        
#        """
#        self.__SaveSerializedData (self.dictBigrams,self.bidictFilename)
##        with open('dictBigrams.pkl', 'wb') as f:
##            dill.dump(self.dictBigrams, f)
##            print ('Bigrams dict saved')
#
#    def LoadDictBigrams (self):
#        """ 
#            This method load the dectionary bigrams for valid words
#            
#        """
#        self.dictBigrams = self.__LoadSerializedData(self.bidictFilename)
#        try:
#            with open('dictBigrams.pkl', 'rb') as f:
#                self.dictBigrams = dill.load(f)
#            self.len_dict_bigrams = len(self.dictBigrams)
#            print ('dict bigrams loaded')
#        except:
#            return False
#        
#==============================================================================
# end new 
#==============================================================================


#
#    def SaveDictBigrams (self):
#        with open('dictBigrams.pkl', 'wb') as f:
#            dill.dump(self.dictBigrams, f)
#            print ('Bigrams dict saved')
#
#    def LoadDictBigrams (self):
#        try:
#            with open('dictBigrams.pkl', 'rb') as f:
#                self.dictBigrams = dill.load(f)
#            self.len_dict_bigrams = len(self.dictBigrams)
#            print ('dict bigrams loaded')
#        except:
#            return False
#
#    def SaveVectorsOut (self, vectors):
#        with open('vectorsOut.pkl', 'wb') as f:
#            dill.dump(vectors, f)
#            
#    def LoadVectorsOut (self):
#        try:
#            with open('vectorsOut.pkl', 'rb') as f:
#                return dill.load(f)
#        except:
#            return False
#
#    def SaveVectors (self, vectors):
#        try:
#            with open('vectors.pkl', 'wb') as f:
#                dill.dump(vectors, f)
#        except:
#            print ("error")
#    def LoadVectorss (self):
#        try:
#            with open('vectors.pkl', 'rb') as f:
#                return dill.load(f)
#        except:
#            return False


if __name__ == '__main__':
    a = CreateBigrams()
    txt = 'prova'
    print (a.Word2Bigrams(txt))

    txt='questa prova'.split()
    print (a.Words2Bigrams(txt))
    print
    print (a.UniqBigrams(txt))
    print ('-'*50)
    print (a.DictVecBigrams(txt))
    print (a.VectorizeWord('prova'))
