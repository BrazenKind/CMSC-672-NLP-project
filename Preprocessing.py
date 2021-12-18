#!/usr/bin/env python
# coding: utf-8

# In[5]:


import regex as re


# In[6]:


class Preprocessing:
    
    def __init__(self, covid_tokens, stop_tokens):
        self.c_t = covid_tokens
        self.s_t = stop_tokens
        
    def clean_text(self, text):
        no_amp_text = re.sub('&amp;', '', text)
        tokenized_words = [re.sub('[^a-zA-Z0-9-]', '', t) for t in no_amp_text.split()]

        return list(filter(None, tokenized_words))
        
    def scrape_sentences(self, text):
        sentences = re.split(self.s_t, text)
        returned_sentences = []
        
        for s in sentences:
            if re.search(self.c_t, s):
                returned_sentences.append(self.clean_text(s))
                
        return returned_sentences  
        


# In[ ]:




