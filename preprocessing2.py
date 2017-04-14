# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:20:50 2017

@author: Patrizio Bellan
         CIMeC
         Università degli studi di Trento
"""


import CreateBigrams

print("Transforming data into proper format")
bi = CreateBigrams.CreateBigrams()
vectors = bi.LoadVectorss()
print ('vecotors loaded')


# permutation and combination
import itertools

lst=list(itertools.combinations(['a','b','c'],2))

print ("CSV saving \n TODO")

print ("transformation in tensors")