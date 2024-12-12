# All required libraries are imported here for you.
import pandas as pd
import numpy as np
import seaborn as sns

#Load the dataset
blood = pd.read_csv('blood.csv')
blood = blood.rename(columns = {'Recency (months)': 'months since last donation',
                              'Frequency (times)': 'total number of donations', 
                              'Monetary (c.c. blood)': 'total blood donated (in c.c.)',
                             'Time (months)': 'months since first donation', 
                              'whether he/she donated blood in March 2007': 'Donated'})

blood
