# All required libraries are imported here for you.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

np.random.seed(123)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.33)

X1_train, X1_temp, y1_train, y1_temp = train_test_split(X1, y1, test_size = 0.3)
X1_valid, X1_test, y1_valid, y1_test = train_test_split(X1_temp, y1_temp, test_size = 0.33)
