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
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import set_config

# Predict DT on train data
dt_pred_train = DTmodel.predict(X_train[features])

# Predict DT on validation data
dt_pred_valid = DTmodel.predict(X_valid[features])

# Predict DT on test data
dt_pred_test = DTmodel.predict(X_test[features])

# Predict NB on train data
nb_pred_train = NBmodel.predict(X_train[features])

# Predict NB on validating data
nb_pred_valid = NBmodel.predict(X_valid[features])

# Predict NB on test data
nb_pred_test = NBmodel.predict(X_test[features])

# Predict knn on train data
knn_pred_train = knn_fit.predict(X_train)

# Predict knn on validating data
knn_pred_valid = knn_fit.predict(X_valid)

# Predict knn on test data
knn_pred_test = knn_fit.predict(X_test)
