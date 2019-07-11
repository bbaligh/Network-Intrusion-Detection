# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:25:22 2019

@author: b.baligh@gmail.com
"""

# %% Imports

from sklearn.model_selection import train_test_split

from pylab import rcParams

import os

import seaborn as sns 

from cnn_lib import (load_datasets, binarize_labels, build_model, 
                     fit_model, normalize_data, eval_model, plot_accuracy_loss,
                     load_model_from, plot_roc_auc, plot_confusion_matrix)

RANDOM_SEED = 42

# %% Main function
def main(in_path=None,
         out_path=None):
    
    if in_path == None:
        in_path = ''
    if out_path == None:
        out_path = ''
   
    sns.set(style='whitegrid', palette='muted', font_scale=1.5) 
    rcParams['figure.figsize'] = 14, 8
    
    X, Y = load_datasets(in_path+'X.csv', 
                         in_path+'y.csv', 
                         verbose_mode=1)

    # %% 
    yy, labels = binarize_labels(Y)    
    
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
    if os.path.isfile(out_path+'cnn2.h5'):
        os.remove(out_path+'cnn2.h5')
    history, _ = fit_model(cnn_1D, 
                           x_train, 
                           x_test, 
                           y_train, 
                           y_test, 
                           checkpointer_file=out_path+'cnn2.h5',
                           nb_epoch=5,
                           batch_size=100,
                           verbose_mode=1)
    
    # %%
    cnn_1D = load_model_from(out_path+'cnn2.h5')
    
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
                       saveto_file=out_path+'loss_accuracy.png',
                       plot_acc=True,
                       plot_loss=True)
    
    # %%
    print("* Prediciting test subset...")
    y_score = cnn_1D.predict(x_test, verbose=1)
    
    # %%
    
    # %%    
    plot_roc_auc(y_test,
                 y_score,
                 labels=labels,
                 saveto_file=out_path+'roc_auc.png')
    # %%
    plot_confusion_matrix(y_test,
                          y_score,
                          labels=labels,
                          title='Confusion matrix, without normalization',
                          saveto_file=out_path+'cm_without_norm.png')
    # %%
    plot_confusion_matrix(y_test,
                          y_score,
                          labels=labels, normalize=True,
                          title='Normalized confusion matrix',
                          saveto_file=out_path+'cm_with_norm.png')# %%
    # %%
    sns.set(style='whitegrid', palette='muted', font_scale=1) 
    rcParams['figure.figsize'] = 14, 8
    plot_confusion_matrix(y_test,
                          y_score,
                          labels=labels, 
                          mixed=True,
                          title='Mixed confusion matrix',
                          saveto_file=out_path+'cm_mixed.png')
    
    return

# %%
if __name__ == '__main__':    
    main()