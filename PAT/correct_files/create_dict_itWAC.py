#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 01:04:44 2017

@author: patrizio
"""

import dill

from Word2Vec import Word2Bigrams, Pos2Vec


def SaveSerializedData (self, data, filename):
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
    
def LoadSerializedData (self, filename):
    try:
        with open(filename, 'rb') as f:
            return dill.load(f)
    except:
        return False

fine='itWAC_fine'
gain='itWAC_gain'

w2v = Word2Bigrams()

dic = {}
with open(gain,'r') as f:
    for line in f.readlines():
        line = line.split()
        # check that the line is correct
        if len(line) == 2:
            # check if it is a valid word
            if w2v._ValidateWord(line[0]):
                #check if the word is already present in the dict
                if line[0] in dic.keys():
                    if line[1] != dic[line[0]]:
                        # there is the key but with a different value
                        print (line[0],line[1],'already present with pos',dic[line[0]])
                else:
                    dic[line[0]] = line[1]
