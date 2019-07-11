# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:16:02 2019

@author: b.baligh@gmail.com
"""

# %% imports
import numpy as np

import seaborn as sns 
from pylab import rcParams 
import ds_tools

#%% 
#Global variables
dataset_path = "C:\\IDSwDL\\Dataset\\Nouveau dossier"

# %% Main
if __name__ == '__main__':
    
    sns.set(style='whitegrid', palette='muted', font_scale=1.5) 
    rcParams['figure.figsize'] = 14, 8 
    RANDOM_SEED = 42 
    LABELS = []
    
# %% 
    df, c, r, nb = ds_tools.load_all(dataset_path, wildcard='Fri*.csv')
# %%
    print("\n\n* Dastaset information") 
    print("  Total dateset files: {}".format(nb))
    print("  Dataset columns: {}".format(c))
    print("  Dataset rows: {}".format(r))
    print("  Dataset head")
    #df.head()
    #df.info()
    
# %%
    # Checking for Null values
    print("\n\n* Checking for null values")
    print("  Null values: {}".format(df.isnull().values.any()))    
    print("  # of Null values: {}".format(df.isnull().sum().sum())) 
# %%    
    # Replacing nan values by 0
    print("\n\n* Replacing nan values by 0") 
    df = df.replace(np.nan, 0, regex=True)
    print("  # of Null values: {}".format(df.isnull().sum().sum())) 
# %% 
    # Replacing nan values by column's max
    for i in range(c-1):
        df[df.columns[i]] = df[df.columns[i]].astype('float64')
        m = df.loc[df[df.columns[i]] != np.inf, df.columns[i]].max()
        df[df.columns[i]].replace(np.inf,m,inplace=True)
        print("  {0} {1:<24s} max= {2}".format(i, df.columns[i], m))
    df.info()
# %%
    # Saving dataframe to csv file
    df.to_csv(dataset_path+'\\corrected_dataset.csv', index=False)
    df = ''