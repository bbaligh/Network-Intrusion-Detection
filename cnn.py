# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:25:22 2019

@author: b.baligh@gmail.com
"""

# %% Imports

from sklearn.model_selection import train_test_split

from pylab import rcParams

from misc_func import (secondsToStr, load_datasets, binarize_labels, build_model, 
                       fit_model, normalize_data, eval_model, plot_accuracy_loss,
                       load_model_from)

RANDOM_SEED = 42

   
# %% Main
if __name__ == '__main__':    
    rcParams['figure.figsize'] = 14, 8
    
    X, Y = load_datasets('x_encoded.csv', 
                         'labeled.csv', 
                         verbose_mode=1)

    # %% 
    yy = binarize_labels(Y)    
    
    # %% 
    # split dataframe to train and test 
    print("* Splitting the data (train & test)...")
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        yy, 
                                                        test_size=0.2, 
                                                        random_state=RANDOM_SEED)
        
    # %%
    #Normalized training set and test set
    x_train, x_test = normalize_data(x_train, x_test)
        
    # %%
    cnn_1D = build_model(n_features=x_train.shape[1],
                         n_outputs=y_train.shape[1],
                         n_kernel=7,
                         n_filters=64,
                         verbose_mode=1)
   
    # %%
    history, _ = fit_model(cnn_1D, 
                           x_train, 
                           x_test, 
                           y_train, 
                           y_test, 
                           checkpointer_file="cnn1.h5",
                           nb_epoch=5,
                           batch_size=100,
                           verbose_mode=1)
    
    # %%
    cnn_1D = load_model_from('cnn1.h5')
    
    # %%
    _, _, _, _ = eval_model(cnn_1D,                            
                            x_train, 
                            x_test, 
                            y_train, 
                            y_test, 
                            eval_train=True,
                            eval_test=True,
                            batch_size=100,
                            verbose_mode = 1)
    # %%
    
    plot_accuracy_loss(history, 
                       saveto_file='loss_accuracy.png',
                       plot_acc=True,
                       plot_loss=True)
    