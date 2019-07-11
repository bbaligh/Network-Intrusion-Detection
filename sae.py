# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:56:06 2019

@author: b.baligh@gmail.com
"""
# %% Imports

from sklearn.model_selection import train_test_split
import pandas as pd 

import seaborn as sns 
from pylab import rcParams 

from sae_lib import (load_ds, display_stats, display_class_stats, normalize_data,
                     X_y_split, train_test_normalize, binarize_labels,
                     build_sae, fit_sae, eval_sae, load_model_from,
                     plot_loss, plot_accuracy)

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
    # split dataframe to train and test 
        # Preparing the data
    print("\n\n* Splitting the data (train & test)...")
    #train, test = train_test_split(df, col_true, stratify=col_true, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=RANDOM_SEED)

# %%    
    x_train, x_test = train_test_normalize(x_train, 
                                           x_test, 
                                           verbose_mode=1)
    
# %%    
    y_train, y_test = binarize_labels(y_train, y_test)
# %% 
    SAE, encoder, _ = build_sae(input_dim=x_train.shape[1], # 78                
                                hidden1_dim=64,                    
                                hidden2_dim=50,                     
                                encoding_dim=25, ## 5x5 for the next CNN network
                                verbose_mode=1)    
# %%
#    from ann_visualizer.visualize import ann_viz;

#    ann_viz(SAE, title="My first neural network")
# %% 
    history, _ = fit_sae(SAE, 
                         x_train, 
                         x_test, 
                         y_train, 
                         y_test, 
                         checkpointer_file="sae.h5",
                         nb_epoch=20,
                         batch_size=320,
                         verbose_mode=1)
        
# %% 
    # And load the saved model (just to check if it works):
    SAE = load_model_from('sae.h5')
# %% 
    """
    predictions = SAE.predict(x_test)
    for ii in range(78):
        plt.title('predict var_{} ({})'.format(ii, df.columns[ii]))
        plt.scatter(x_test[:, ii], predictions[:, ii])
        plt.savefig('predict_var_{}.png'.format(ii));
        plt.close()
        """        
# %%
    _, _, _, _ = eval_sae(SAE,
                          x_train,
                          x_test,
                          eval_train = True,
                          eval_test = True,
                          batch_size = 32,
                          verbose_mode = 1)
    
# %%
    plot_loss(history, 
              saveto_file='model_loss.png')    
    
# %%
    plot_accuracy(history,               
                  saveto_file='model_accuracy.png')
    
# %%    
    # X encoding
    print("\n\n* Encoding dataset file")
    x_encoded = encoder.predict(X)
# %%
    # Saving x_encoded to a CSV file
    print("\n\n* Saving encoded and labeled dataset files")
    pd.DataFrame(x_encoded).to_csv('x_encoded.csv', index=False)
    pd.DataFrame(y).to_csv('labeled.csv', index=False)
    df = ''