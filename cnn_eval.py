# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:23:11 2019

@author: b.baligh@gmail.com
"""
# evaluation of cnn model with filters
from sklearn.model_selection import train_test_split

from pylab import rcParams

from numpy import mean
from numpy import std
import matplotlib.pyplot as plt1

from misc_func import (secondsToStr, load_datasets, binarize_labels, build_model, 
                       fit_model, normalize_data, eval_model, plot_accuracy_loss,
                       load_model_from)

RANDOM_SEED = 42

#******************************************************************************
# fit and evaluate a model
def evaluate_model(trainX, 
                   trainY, 
                   testX, 
                   testY, 
                   nkernel = 3,
                   nfilters = 64,
                   nbatch_size = 32,
                   nepochs = 10,
                   nverbose_mode = 0):
    # build the model
    cnn_eval_f = build_model(n_features=trainX.shape[1], 
                             n_outputs=trainY.shape[1],
                             n_kernel=nkernel,
                             n_filters=nfilters,
                             verbose_mode=nverbose_mode)
    # fit network
    history, _ = fit_model(cnn_eval_f,
                           trainX,
                           testX,
                           trainY,
                           testY,
                           checkpointer_file="cnn_eval_f.h5",
                           nb_epoch=nepochs,
                           batch_size=nbatch_size,
                           verbose_mode=nverbose_mode)
    # evaluate model
    accuracy, _,  = eval_model(cnn_eval_f,                            
                               trainX, 
                               testX, 
                               trainY, 
                               testY, 
                               eval_train=False,
                               eval_test=True,
                               batch_size=nbatch_size,
                               verbose_mode=nverbose_mode)
    return accuracy

#******************************************************************************
# summarize scores
def summarize_results(scores, 
                      params,
                      chart_path):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
    plt1.boxplot(scores, labels=params)
    plt1.savefig(chart_path)
    return

#******************************************************************************
# run a filter experiment 
def run_filter_experiment(trainX,                    
                          trainY, 
                          testX, 
                          testY, 
                          params, 
                          repeats = 10):
    # test each parameter
    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = evaluate_model(trainX, 
                                   trainY, 
                                   testX, 
                                   testY, 
                                   nfilters=p,
                                   nepochs=3,
                                   nverbose_mode=0)  
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r+1, score))
            scores.append(score)
        all_scores.append(scores)
    # summarize results
    summarize_results(all_scores, params, 'cnn_filters.png')
    return

#******************************************************************************
# run a kernel experiment 
def run_kernel_experiment(trainX,                    
                          trainY, 
                          testX, 
                          testY, 
                          params, 
                          repeats = 10):
    # test each parameter
    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = evaluate_model(trainX, 
                                   trainY, 
                                   testX, 
                                   testY, 
                                   nepochs=3,
                                   nkernel=p,
                                   nverbose_mode=0)  
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r+1, score))
            scores.append(score)
        all_scores.append(scores)
    # summarize results
    summarize_results(all_scores, params, 'cnn_kernels.png')
    return

#******************************************************************************
# %% Main
if __name__ == '__main__':    
    rcParams['figure.figsize'] = 14, 8
    
    X, Y = load_datasets('x_encoded.csv', 
                         'labeled.csv', 
                         verbose_mode=1)
    # %% 
    # binarizing labels 
    yy = binarize_labels(Y)    
    # %% 
    # split dataframe to train and test 
    print("* Splitting the data (train & test)...")
    x_train, x_test, y_train, y_test = train_test_split(X, 
                                                        yy, 
                                                        test_size=0.2, 
                                                        random_state=RANDOM_SEED)
    # %%
    # normalized training set and test set
    x_train, x_test = normalize_data(x_train, x_test)
    # %%
    # run the filter experiments
    n_params = [8, 16, 32, 64]
    run_filter_experiment(x_train,                    
                          y_train, 
                          x_test, 
                          y_test,
                          n_params,
                          repeats=5)
    # %%
    # run the kernel experiments
    n_params = [2, 3, 5, 7, 11]
    run_kernel_experiment(x_train,                    
                          y_train, 
                          x_test, 
                          y_test,
                          n_params,
                          repeats=5)