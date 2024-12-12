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
