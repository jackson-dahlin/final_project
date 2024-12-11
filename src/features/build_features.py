#!/usr/bin/env python
# coding: utf-8

# In[1]:


# All required libraries are imported here for you.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Features
X1 = log_blood[['months since last donation', 'total number of donations', 
           'total blood donated (in c.c.)', 'months since first donation']]

# Labels
y1 = log_blood['Donated']

# Features
X = blood[['months since last donation', 'total number of donations', 
           'total blood donated (in c.c.)']]
features = ['months since last donation', 'total number of donations', 
           'total blood donated (in c.c.)']

# Labels
y = blood['Donated']




