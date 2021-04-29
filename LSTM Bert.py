#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoTokenizer, AutoModel
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


# Normal dataset
df = pd.read_csv("domain_data.csv")


# In[3]:


# Updating values for training_data
training_data = df[df['split'] == 'train']
training_data = training_data.drop(training_data.query('toxicity==0').sample(frac=.85).index)

# Getting test_data
test_data = df[df['split'] == 'test']

# Getting validation_data
validation_data = df[df['split'] == 'val']
validation_data = validation_data.drop(validation_data.query('toxicity==0').sample(frac=.85).index)


# In[4]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# In[5]:


# Creating data loaders
X_train = np.array(training_data['comment_text'].values.tolist())
Y_train = np.array(training_data['toxicity'].values.tolist())

X_test = np.array(test_data['comment_text'].values.tolist())
Y_test = np.array(test_data['toxicity'].values.tolist())

X_val = np.array(validation_data['comment_text'].values.tolist())
Y_val = np.array(validation_data['toxicity'].values.tolist())





def BuildSentenceMatrix(text, embedding_dimension, features):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    outputs = outputs[0].squeeze().detach().numpy()
    if len(outputs) < embedding_dimension:
        zeroVectors = np.zeros((embedding_dimension - len(outputs), features))
        outputs = np.concatenate((outputs, zeroVectors))
    elif len(outputs) > embedding_dimension:
        outputs = outputs[:embedding_dimension]
    return outputs

def ConvertData(data, embedding_dimension, features):
    converted_dataset = []
    for i in range(len(data)):
        converted_dataset.append(BuildSentenceMatrix(data[i], embedding_dimension, features))
    return np.array(converted_dataset)


# In[9]:


transformedTestSet = ConvertData(X_test[:100], 100, 768)


# In[14]:


transformedTestSet.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




