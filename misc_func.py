# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:03:12 2019

@author: b.baligh@gmail.com
"""
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

#from scipy.interpolate import spline

from time import strftime, localtime, time
from datetime import timedelta

#******************************************************************************
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))
    
#******************************************************************************
#Loading dataset
def load_datasets(data_file, 
                  label_file = None,
                  label_col = -1,
                  verbose_mode = 1):
    print("* Loading encoded dataset...")
    df = pd.read_csv(data_file, header=0)
    if label_file != None:
        XX = df.iloc[:,:].values
    else:
        if label_col == -1:
            XX = df.iloc[:,0:-1].values # All but the last column
        else:
            XX = df.iloc[:,0:label_col-1].values
    if verbose_mode != 0:
        print("  Dataset shape: {}".format(XX.shape))
     
    print("* Loading dataset labels...")
    if label_file != None:
        df = pd.read_csv(label_file, header=0)
        YY = df.iloc[:,:].values
    else:
        if label_col == -1:
            YY = df.iloc[:,-1].values # last column
        else:
            YY = df.iloc[:,label_col-1].values
    if verbose_mode != 0:
        print("  Labels shape: {}".format(YY.shape))
    
    return(XX, YY)
    
#******************************************************************************
#Binarizing labels
def binarize_labels(labels,
                    verbose_mode = 1):
    print("* Binarizing labels array...")
    y_labels = pd.DataFrame(labels)[0].unique()
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    
    y = labels.reshape(1,len(labels))
    y = y[0]
    y = le.transform(y)
    y = to_categorical(y)

    if verbose_mode != 0:
        print("Labels: ", y_labels)
    
    return(y)

#******************************************************************************    
#Building a model
def build_model(n_features, 
                n_outputs,
                n_kernel = 5,    
                n_filters = 64,
                verbose_mode = 1): 
    print("* Building the model...")
    """
    cnn_1D=Sequential()
    cnn_1D.add(Conv1D(64,1,activation='relu',input_shape=(25,1)))
    cnn_1D.add(Conv1D(64,1,activation='relu'))
    cnn_1D.add(MaxPooling1D(3))
    cnn_1D.add(Conv1D(64,1,activation='relu'))
    cnn_1D.add(Conv1D(64,1,activation='relu'))
    cnn_1D.add(GlobalAveragePooling1D())
    cnn_1D.add(Dropout(0.5))
    cnn_1D.add(Dense(len(y_labels),activation='softmax'))
    """
    model = Sequential()
    model.add(Conv1D(filters=n_filters, 
                      kernel_size=n_kernel, 
                      activation='relu', 
                      input_shape=(n_features,1)))
    model.add(Conv1D(filters=n_filters, 
                      kernel_size=n_kernel, 
                      activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    """
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=(['accuracy']))
    """
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=(['accuracy']))
    if verbose_mode != 0:
        model.summary()
        
    return(model)

#******************************************************************************
#Fitting the model
def fit_model(model,
              x,
              y,
              x_lb,
              y_lb,
              checkpointer_file="model.h5",
              tensorboard_logdir='./logs',
              nb_epoch = 5,
              batch_size = 32,
              verbose_mode = 1):
    print("* Fitting the network...")
    checkpointer = ModelCheckpoint(filepath=checkpointer_file,
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir=tensorboard_logdir,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    start = time()
    history = model.fit(x, 
                        x_lb, 
                        validation_data=(y, y_lb), 
                        batch_size=batch_size, 
                        epochs=nb_epoch,
                        verbose=verbose_mode,
                        callbacks=[checkpointer, tensorboard])#.history
    fit_time = time()-start
    if verbose_mode != 0:
        print("Training time: ",secondsToStr(fit_time))
        
    return(history, fit_time)

#******************************************************************************    
#Normalizing training set and test set
def normalize_data(train_data, test_data):
    print("* Normalizing the dataset...")
    scaler = Normalizer().fit(train_data)
    train_data = scaler.transform(train_data)
    
    scaler = Normalizer().fit(test_data)
    test_data = scaler.transform(test_data)
    
    train_data = np.expand_dims(train_data, axis=2)
    test_data = np.expand_dims(test_data, axis=2)
    
    return(train_data, test_data)

#******************************************************************************    
def eval_model(model,
               train_data,
               test_data,
               train_label,
               test_label,
               eval_train = True,
               eval_test = True,
               batch_size = 32,
               verbose_mode = 1):
    print("* Evaluating the model...")
    # evaluate the model
    #score=cnn_1D.evaluate(y, y_test, batch_size=batch_size)
    #print('Score: ',score)
    if eval_train:
        print("*** Evaluating the train dataset...")
        train_loss, train_acc = model.evaluate(train_data, 
                                               train_label, 
                                               verbose=verbose_mode)
    if eval_test:
        print("*** Evaluating the test dataset...")
        test_loss, test_acc = model.evaluate(test_data, 
                                             test_label, 
                                             verbose=verbose_mode)
    if verbose_mode != 0:
        if eval_train or eval_test:
            print("       Accuracy      Loss")
        if eval_train:
            print('Train: %8.3f  %8.3f' % (train_acc, train_loss))
        if eval_test:
            print('Test : %8.3f  %8.3f' % (test_acc, test_loss))
    if eval_train and eval_test:
        return(train_acc, train_loss, test_acc, test_loss)
    elif eval_train:
        return(train_acc, train_loss)
    elif eval_test:
        return(test_acc, test_loss)
    else:
        return
    
def load_model_from(model_file):
    print("* Loading saved model...")
    model = load_model(model_file)
    
    return(model)

#******************************************************************************        
def plot_accuracy_loss(hist,
                       saveto_file,
                       plot_acc = True,
                       plot_loss = True):
    # plot loss during training
    if plot_acc:
        if plot_loss:
            plt.subplot(211)
        plt.title('Loss')
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='test')
        plt.legend()
    
    # plot accuracy during training
    if plot_loss:
        if plot_acc:
            plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(hist.history['acc'], label='train')
        plt.plot(hist.history['val_acc'], label='test')
        plt.legend()
    #plt.show()
    if (plot_acc or plot_loss) and (saveto_file != ""):
        plt.savefig(saveto_file)
    return