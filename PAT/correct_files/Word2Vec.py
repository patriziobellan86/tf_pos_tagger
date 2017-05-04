#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:44:51 2017

@author: patrizio
"""

from itertools import permutations
import morphItDataExtractor

class VectorizeBigrams:
    letters = ['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t','u','v','z','j','k','x','y','w','#']
    
    def __init__(self):#, letters = None):
#        self.letters = letters or self.letters
        self._CreateBiDict()
        
    def _CreateBiDict(self):
        l = list(permutations(self.letters, 2))
        bi = [a+b for (a,b) in l]
        bi=set(bi)
        for l in self.letters:
            bi.add((l+l))
        bi=list(bi)
        bi.sort()
        bi=bi[1:]    
        self.bidict = {k:i for i,k in enumerate(bi)}
    
    def _VoidBiDict(self):
        return {k:0 for k in self.bidict.keys()}
    
    def _CreateVector (self, bigrams):
        """
            args:
                bigrams
                    dict (bigram: numer_of_this_bigram)
        """
        vector = [0]*len(self.bidict)
        for bi in bigrams.keys():
            vector[self.bidict[bi]]=bigrams[bi]
        return vector


class Word2Bigrams (VectorizeBigrams):
    """
        This class create bigrams
    """
    start_tag='#'
    end_tag='#'
#    letters = ['a','b','c','d','e','f','g','h','i','l','m','n','o','p','q','r','s','t','u','v','z','j','k','x','y','w','#']
    
    def __init__(self):
        super(Word2Bigrams,self).__init__()
        self.validchars = self.letters[:-1]
        
    def _ValidateWord (self, word):
        if [False for l in word if l not in self.validchars]:
            return False
        return True
    
    def _Word2Bigrams (self, word):
        bigrams=self._VoidBiDict()
        
        bigrams[self.start_tag+word[0]] = 1
            
        # main loop
        for i in range(len(word)-1):
                bigrams[word[i]+word[i+1]]=bigrams[word[i]+word[i+1]]+1
        # last char
        bigrams[word[-1]+self.end_tag]=1
        
        return bigrams
    
    def Word2Vec (self, word):
        word = word.lower()
        if not self._ValidateWord(word):
            return False
        return self._CreateVector(self._Word2Bigrams(word))

# pos tag taken from morphit
class Pos2Vec:
    def __init__(self):
        self.words = morphItDataExtractor.MorphItDataExtractor('morphitUtf8.txt').words
        self._Pos2Dict()
        
    def _Pos2Dict (self):
        pos = set(self.words.values())
        pos = list(pos)
        pos.sort()
        
        self.posdict = {k:i for i,k in enumerate(pos)}    
        
    def Pos2Vec(self, pos):
        vector = [0]*len(self.posdict)
        vector[self.posdict[pos]] = 1
        return vector
    
    def PosFromIndex(self, index):
        return list(self.posdict.keys())[list(self.posdict.values()).index(index)]
        
if __name__ == '__main__':
    w='cacacacasa'
    print (Word2Bigrams().Word2Vec(w))

    pos = 'ADJ'
    print (Pos2Vec().Pos2Vec(pos))
#    
#    bivec=a.Word2Bigrams(w)
#    print (bivec)
#    
#    vec = a.CreateVector(bivec)
#    print (vec)
#    
##    
#    
#    a = VectorizeBigrams()
#    w = {'#a':1, '#b':2,'zy':55,'zz':27, 'ai':304}
#    v = a.CreateVector(w)
#    print (v)