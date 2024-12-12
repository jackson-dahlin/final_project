# All required libraries are imported here for you.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create the log dataframe
log_blood = blood.copy()

# Fix values <= 0 by replacing them with NaN
log_blood['months since last donation'] = log_blood['months since last donation'].apply(lambda x: np.nan if x <= 0 else x)
log_blood['total number of donations'] = log_blood['total number of donations'].apply(lambda x: np.nan if x <= 0 else x)
log_blood['total blood donated (in c.c.)'] = log_blood['total blood donated (in c.c.)'].apply(lambda x: np.nan if x <= 0 else x)
log_blood['months since first donation'] = log_blood['months since first donation'].apply(lambda x: np.nan if x <= 0 else x)

# Take the logarithm of the fixed values
log_blood['months since last donation'] = np.log10(log_blood['months since last donation'])
log_blood['total number of donations'] = np.log10(log_blood['total number of donations'])
log_blood['total blood donated (in c.c.)'] = np.log10(log_blood['total blood donated (in c.c.)'])
log_blood['months since first donation'] = np.log10(log_blood['months since first donation'])

# Fix values NaN by replacing them with 0
log_blood['months since last donation'] = log_blood['months since last donation'].fillna(0)
log_blood['total number of donations'] = log_blood['total number of donations'].fillna(0)
log_blood['total blood donated (in c.c.)'] = log_blood['total blood donated (in c.c.)'].fillna(0)
log_blood['months since first donation'] = log_blood['months since first donation'].fillna(0)

log_blood
