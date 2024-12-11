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

# Creating a confusion matrix for the dt validating data
dt_tn, dt_fp, dt_fn, dt_tp = confusion_matrix(y_valid, dt_pred_valid).ravel()
dt_cm = confusion_matrix(y_valid, dt_pred_valid)

# Creating a confusion matrix for the knn validating data
knn_tn, knn_fp, knn_fn, knn_tp = confusion_matrix(y_valid, knn_pred_valid).ravel()
knn_cm = confusion_matrix(y_valid, knn_pred_valid)

# Creating a confusion matrix for the nb validating data
nb_tn, nb_fp, nb_fn, nb_tp = confusion_matrix(y_valid, nb_pred_valid).ravel()
nb_cm = confusion_matrix(y_valid, nb_pred_valid)

# Evaluate the model on dt valdiating data
dt_accuracy = metrics.accuracy_score(y_valid, dt_pred_valid)
dt_precision = precision_score(y_valid, dt_pred_valid)
dt_recall = metrics.recall_score(y_valid, dt_pred_valid)
dt_specificity = dt_tn / (dt_tn + dt_fp)
dt_f1 = metrics.f1_score(y_valid, dt_pred_valid)

# Evaluate the model on knn valdiating data
knn_accuracy = metrics.accuracy_score(y_valid, knn_pred_valid)
knn_precision = precision_score(y_valid, knn_pred_valid)
knn_recall = metrics.recall_score(y_valid, knn_pred_valid)
knn_specificity = knn_tn / (knn_tn + knn_fp)
knn_f1 = metrics.f1_score(y_valid, knn_pred_valid)

# Evaluate the model on nb valdiating data
nb_accuracy = metrics.accuracy_score(y_valid, nb_pred_valid)
nb_precision = precision_score(y_valid, nb_pred_valid)
nb_recall = metrics.recall_score(y_valid, nb_pred_valid)
nb_specificity = nb_tn / (nb_tn + nb_fp)
nb_f1 = metrics.f1_score(y_valid, nb_pred_valid)
