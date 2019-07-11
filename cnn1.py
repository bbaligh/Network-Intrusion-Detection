# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:39:35 2019

@author: hp
"""

# %% Imports

from sklearn.model_selection import train_test_split
import pandas as pd 

import seaborn as sns 
from pylab import rcParams 

from sae_lib import (load_ds, display_stats, display_class_stats, normalize_data,
                     X_y_split)

#%% 
#Global variables
dataset_path = "C:\\IDSwDL\\Dataset"

# %% Main
if __name__ == '__main__':
    
    sns.set(style='whitegrid', palette='muted', font_scale=1.5) 
    rcParams['figure.figsize'] = 14, 8 
    RANDOM_SEED = 42 
    
# %% 
    df, c, _ = load_ds(dataset_path, verbose_mode=1)
    
# %% 
    display_stats(df)

# %% 
    """Countinng Label class frequency and distirbution"""
    display_class_stats(df, plot=True)

# %%
    normalize_data(df, verbose_mode=1)      
        
# %% 
    X, y = X_y_split(df, label_col=c-1)

# %% 
    # Saving X to a CSV file
    print("\n\n* Saving X and y dataset files")
    pd.DataFrame(X).to_csv('X.csv', index=False)
    pd.DataFrame(y).to_csv('y.csv', index=False)
    df = ''
    print("  Ok")