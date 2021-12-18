#!/usr/bin/env python
# coding: utf-8

# In[1]:


import regex as re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import random
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
from Preprocessing import *


# In[76]:


class models:
    
    def __init__(self, train_x, train_y, label_IDs, dev_x = None, dev_y = None, test_x = None, test_y = None):
        
        assert len(train_x) == len(train_y), "ERROR: train_x and train_y are of different length!"
        
        if dev_x is not None and dev_y is not None:
            assert len(dev_x) == len(dev_y), "ERROR: dev_x and dev_y are of different length!"
            
        if test_x is not None and test_y is not None:
            assert len(test_x) == len(test_y), "ERROR: test_x and test_y are of different length!"
        
        
        self.train_x = train_x
        self.train_y = train_y
        self.label_IDs = label_IDs
        
        self.dev_x = dev_x
        self.dev_y = dev_y
           
        self.test_x = test_x
        self.test_y = test_y
        
        
        self.Word2Vec = None
        self.MaxEnt = None
        
        
    def assert_len(self, x, y, split):
        if x is not None and y is not None:
            assert len(x) == len(y), "ERROR: %s_x and %s_y datasets are of different length!"%(split, split)
        
    def set_train_data(self, train_x, train_y):
        
        self.assert_len(train_x, train_y, "train")
        
        self.train_x = train_x
        self.train_y = train_y
        
    def set_test_data(self, test_x, test_y):
        
        self.assert_len(test_x, test_y, "test")
        
        self.test_x = test_x
        self.test_y = test_y
        
    def set_dev_data(self, dev_x, dev_y):
        
        self.assert_len(dev_x, dev_y, "dev")
        
        self.dev_x = dev_x
        self.dev_y = dev_y
        
    
    def train_Word2Vec(self):
        tagged_docs = [TaggedDocument(words = x, tags = str(y)) for x in self.train_x for y in self.train_y]
        
        x_merged = []
        
        for lst in self.train_x:
            x_merged.extend(lst)
        
        x_dict = {x: x_merged.count(x) for x in set(x_merged)}
        model_dm = Doc2Vec(dm = 1, vector_size = 300, epochs = 50)
        model_dm.build_vocab_from_freq(x_dict)
        model_dm.train(tagged_docs, total_examples=len(tagged_docs), epochs=model_dm.epochs)
        
        self.Word2Vec = model_dm
        
    
    def train_MaxEnt(self):
        
        assert self.Word2Vec is not None, "ERROR: Word2Vec model not initialized yet! Call models.train_Word2Vec() to initialize."
        
        try:
            train_x_vectorized = [self.Word2Vec.infer_vector(x) for x in self.train_x]
        except AssertionError:
            print("ERROR: Word2Vec model not initialized yet! Call models.train_Word2Vec() to initialize.")
            return
        
        logreg = LogisticRegression(max_iter = 1000, C=1e5)
        logreg.fit(train_x_vectorized, self.train_y)
        
        self.MaxEnt = logreg
        
    def save_Word2Vec(self):
        return self.Word2Vec
    
    def load_Word2Vec(self, Word2Vec):
        self.Word2Vec = Word2Vec
    
    def predict_sentence(self, sentence_list):
        
        assert self.Word2Vec is not None, "ERROR: Word2Vec model undefined! Train it by calling model.train_Word2Vec()"
        assert self.MaxEnt is not None, "ERROR: MaxEnt model undefined! Train it by calling model.train_MaxEnt()"
        
        sentence_vector = self.Word2Vec.infer_vector(sentence_list).reshape(1, -1)
        predict = self.MaxEnt.predict_proba(sentence_vector)[0]
        
        for i in range(0, len(predict)):
            print("Label %s (%s) probability: "%(i, self.label_IDs[i]))
            print(np.around(predict[i], decimals = 5))
        
    def test_MaxEnt(self, test_x = None, test_y = None, overwrite = True):
        
        self.assert_len(test_x, test_y, "test")
        
        assert self.test_x is not None and self.test_y is not None, "ERROR: test data undefined! Either set test data by calling models.set_test_data(self, test_x, test_y), or specify them as parameters in this method. (test_MaxEnt(self, test_x = test_x, test_y = test_y, overwrite = True/False.)) Set overwrite to True or False to overwrite existing test data."
        
        if test_x is not None and overwrite:
            self.test_x = test_x
        
        if test_y is not None and overwrite:
            self.test_y = test_y
            
        cur_test_x = None
        cur_test_y = None
        
        if overwrite:
            cur_test_x = self.test_x
            cur_test_y = self.test_y
        else:
            cur_test_x = test_x
            cur_test_y = test_y
            
        predicted = []
        actual = cur_test_y
        
        for x in range(0, len(cur_test_x)):
            #print(logreg.predict_proba(model_dm.infer_vector(x).reshape(1, -1)))
            predict = self.MaxEnt.predict(self.Word2Vec.infer_vector(cur_test_x[x]).reshape(1, -1))[0]                         
            predicted.append(predict)
            
        total_labels = set(cur_test_y)
        
        for label in total_labels:
            print("Stats for label %s (%s): " %(label, self.label_IDs[label]))
            print()
            
            TP = 0
            FP = 0
            FN = 0
            
            for i in range(0, len(predicted)):
                
                predicted_ID = int(predicted[i])
                      
                if predicted_ID == label:
                    if predicted_ID == actual[i]:
                        TP += 1
                    else:
                        FP += 1
                elif predicted_ID == label:
                    FN += 1
            
            #print('TP: ', TP)
            #print('FP: ', FP)
            #print('FN: ', FN)
            
            precision = 0
            recall = 0
            F1 = 0
            
            if TP is not 0:
                
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                F1 = 2*(precision*recall)/(precision+recall)

            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", F1)
            print("_______________________________")
                
            
        
    


# In[68]:


#train_x_vectorized = [model_dm.infer_vector(x) for x in train_x]


# In[69]:


# logreg = LogisticRegression(max_iter = 1000, C=1e5)
# logreg.fit(train_x_vectorized, train_y)


# In[72]:


# for x in range(0, len(test_x)):
#     #print(logreg.predict_proba(model_dm.infer_vector(x).reshape(1, -1)))
#     predict = logreg.predict(model_dm.infer_vector(test_x[x]).reshape(1, -1))[0]
#     actual = test_y[x]
                             
#     print(predict, ' ', actual)

