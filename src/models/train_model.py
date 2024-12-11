#!/usr/bin/env python
# coding: utf-8

# In[2]:


#All required libraries are imported here for you.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import set_config
from sklearn.naive_bayes import GaussianNB

depth_limit = 5

# Create the DT model
DTmodel = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth_limit)

# Fit the DT model
DTmodel.fit(X_train[features], y_train)

# Initialize the NB model
model = GaussianNB()

# Fit the NB model
NBmodel.fit(X_train[features], y_train)

# Create the KNN model
knn_spec = KNeighborsClassifier(n_neighbors = 5)

# Fit the KNN model
knn_fit = knn_spec.fit(X_train, y_train)
