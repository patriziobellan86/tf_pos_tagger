#-*- encoding:utf-8 -*-
"""
                               morphitDataExtractor.py

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
"""
from __future__ import unicode_literals, with_statement, division
import dill
import codecs


class MorphItDataExtractor(object):

    """
    This class extracts and elaborates data from morphit

    """
    def __init__(self, morphit):
        """
            __init__ needs only to knwo where is lexicon
        """
        self.MorphItFileName = morphit

        self.verbTags = ['VER', 'AUX', 'MOD', 'CAU', 'ASP']
        self.words = self.LoadWords()
        if not self.words:
            self.ReadMorphit()
            self.SaveWords()
#        else:
#            print('data loaded')

    def isVerb(self, word):
        if self.words[word] in self.verbTags:
            return 1#True 
        return 0#False
        
    def Words(self):
        return self.words.keys()

    def ReadMorphit(self):
        """
        This method loads all the words from morphit
        """
        self.words = {}
        with codecs.open(self.MorphItFileName, 'r', 'utf-8') as f:
            for line in f.readlines():
                line = line.split()
                try:
#                    print (line)
                    self.words[line[0]] = line[2][:3]
#                    if line[2][:3] in self.verbTags:
#                        line[2]=line[2].split(u'+')
#                        line[2][0]=line[2][0][line[2][0].find(u':')+1:]
                except:
                    pass
        return self.words

    def SaveWords(self):
        with open('mWords.pkl', 'wb') as f:
            dill.dump(self.words, f)

        
    def LoadWords(self):
        try:
            with open('mWords.pkl', 'rb') as f:
                return dill.load(f)
        except:
            return False


if __name__ == '__main__':
    a = MorphItDataExtractor('morphitUtf8.txt')
