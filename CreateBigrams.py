# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:17:14 2016

@author: patrizio
"""

import numpy as np
import dill

class CreateBigrams (object):
    """
        This class create bigrams
    """
    start_tag='#'
    end_tag='#'

    def __init__(self):
        self.dictBigrams = self.LoadDictBigrams()

    def FilterWords (self, words):
        return [w for w in words if w.isalpha()]

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
#            print (word, bigrams)
            return bigrams
        except:
            print (word)
            print(word[i]+word[i+1])

    def Words2Bigrams (self, words):
        bigrams=[]
        for word in words:
#            bigrams.append(self.Word2Bigrams (word))
            bigrams.extend(self.Word2Bigrams (word))
        return bigrams
#    def UniqBigrams (self, bigrams):

    def UniqBigrams (self, words):
        bigrams = self.Words2Bigrams(words)

        uniq = set()
        for bis in bigrams:
            uniq.add(bis)
        return list(uniq)

    def DictVecBigrams (self, words):
        """
            dictBigrams lo uso per passare un eventuale stesso dict e non calcolarlo
        """
        if not self.dictBigrams:
            print ('creazione di dict bigrmas')
            dictBigrams = {bi:n for n,bi in enumerate(self.UniqBigrams(words))}
            self.dictBigrams = dictBigrams
            self.SaveDictBigrams()
            self.len_dict_bigrams = len(self.dictBigrams)
        return self.dictBigrams

    def VectorizeWords(self, words):
#        return [self.VectorizeWord(w) for w in words]
        t=[]
        for n,w in enumerate(words):
            print (n,'/',len(words),w)
            t.append(self.VectorizeWord(w))
        return t
        
    def VectorizeWord(self, word):
        vector = np.zeros((1,self.len_dict_bigrams), dtype=np.int)
        
        for bi in self.Word2Bigrams(word):
            vector[0,self.dictBigrams[bi]] = 1
        return vector


    def SaveDictBigrams (self):
        with open('dictBigrams.pkl', 'wb') as f:
            dill.dump(self.dictBigrams, f)
            print ('Bigrams dict saved')

    def LoadDictBigrams (self):
        try:
            with open('dictBigrams.pkl', 'rb') as f:
                self.dictBigrams = dill.load(f)
            self.len_dict_bigrams = len(self.dictBigrams)
            print ('dict bigrams loaded')
        except:
            return False

    def SaveVectorsOut (self, vectors):
        with open('vectorsOut.pkl', 'wb') as f:
            dill.dump(vectors, f)
            
    def LoadVectorsOut (self):
        try:
            with open('vectorsOut.pkl', 'rb') as f:
                return dill.load(f)
        except:
            return False

    def SaveVectors (self, vectors):
        try:
            with open('vectors.pkl', 'wb') as f:
                dill.dump(vectors, f)
        except:
            print ("error")
    def LoadVectorss (self):
        try:
            with open('vectors.pkl', 'rb') as f:
                return dill.load(f)
        except:
            return False


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
