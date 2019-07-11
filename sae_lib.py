# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:53:29 2019

@author: b.baligh@gmail.com
"""
import pandas as pd 
import numpy as np
import os
#import glob
#import shutil
#import math
from sklearn.preprocessing import MinMaxScaler

from keras.models import Model, load_model 
from keras.layers import Input, Dense 
from keras.callbacks import ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt

from time import time
from misc_func import secondsToStr
import ds_tools

#******************************************************************************        
# Load & Display minimum info about all dataset files Stats located in dataset_path
def load_ds(dataset_path, 
            verbose_mode = 0):
    df, cols, rows, _ = ds_tools.load_all(dataset_path)
    if verbose_mode != 0:
        print("{} rows".format(rows))
        print("{} cols".format(cols))
        
    return(df, cols, rows)

#******************************************************************************        
# Display all dataset files Stats located in dataset_path
def display_stats(df):
    cols = df.columns.values
    total = float(len(df))
    print("{} rows".format(int(total)))
    print("{} cols".format(len(cols)))
    print("\n\n* Dataset header (5 rows)")
    print(df.head())
    print("\n\n* Dataset summary")
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,ds_tools.expand_categories(df[col])))
            ds_tools.expand_categories(df[col])
    return

#******************************************************************************        
def display_class_stats(df, 
                        plot = True):  
    #LABELS = []

    print("\n\n* Counting normal activities & attacks...")
    attacks = df[df['Label'] != "BENIGN"] 
    normal = df[df['Label'] == "BENIGN"] 
    if plot:
        plt.pie([normal.shape[0], attacks.shape[0]], 
                labels=['normal','attack'], 
                autopct='%1.1f%%', 
                shadow=True, 
                startangle=90)
        plt.title("normal/attack distribution")
        if os.path.isfile('normal_attack_distribution.png'):
            os.remove('normal_attack_distribution.png')
        plt.savefig('normal_attack_distribution.png')
        plt.close()

        #LABELS = df['Label'].unique()
        print("\n\n* Counting Label class frequency and distribution...")
        count_classes = pd.value_counts(df['Label'], sort = True)
        count_classes.plot(kind = 'bar', rot=0)
        plt.title("'Label' class distribution")
        #plt.xticks(range(5), LABELS)
        plt.xlabel("'Label' class")
        plt.ylabel("Frequency")
        if os.path.isfile('label_class_distribution.png'):
            os.remove('label_class_distribution.png')
        plt.savefig('label_class_distribution.png')
        plt.close()
    print("  attacks:",attacks.shape)
    print("  normal:",normal.shape)
    
#******************************************************************************        
def normalize_data(df, 
                   verbose_mode = 1):
    # Preparing the data
    print("\n\n* Normalizing the data...")
    for i in range(df.shape[1]-1):
        if verbose_mode != 0:
            print("\r  Reshaping column: {}".format(i), end="")
        #df[df.columns[i]] = StandardScaler().fit_transform(df[df.columns[i]].values.reshape(-1, 1))
        df[df.columns[i]] = MinMaxScaler().fit_transform(df[df.columns[i]].values.reshape(-1, 1))

#******************************************************************************        
def X_y_split(df, 
              label_col = -1):
    print("\n\n* Preparing the data...")
    # Y = last column contains labels
    y = df.iloc[:,label_col].values 
    # X = all of the dataframe except the last column which is the labels column
    X = df.iloc[:,0:label_col].values
    return(X, y)
    
#******************************************************************************        
def train_test_normalize(x_train, 
                         x_test, 
                         verbose_mode = 1):
    # normalize all values between 0 and 1 
    x_train = (x_train.astype('float64')+1) / 2.
    x_test = (x_test.astype('float64')+1) / 2.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    if verbose_mode != 0:
        print("  Training set: {}".format(x_train.shape[0]))
        print("  Testing set: {}".format(x_test.shape[0]))
    return(x_train, x_test)
    
#******************************************************************************        
def binarize_labels(y_train, y_test):
    # binary classification
    y_train[y_train == "BENIGN"] = 0
    y_train[y_train != 0] = 1
    y_test[y_test == "BENIGN"] = 0
    y_test[y_test != 0] = 1
    # convert values from object to float64
    y_train=y_train.astype('float64')
    y_test=y_test.astype('float64')
    return(y_train ,y_test)

#******************************************************************************        
def build_sae(input_dim = 78,    
              hidden1_dim = 64,    
              hidden2_dim = 50,     
              encoding_dim = 25, 
              verbose_mode = 1):
        # Building the model for the model
    print("\n\n* Building the model...")
    
    if verbose_mode != 0:
        compression_factor = float(input_dim) / encoding_dim
        print("  Compression factor: %s" % compression_factor)

    # "encoded" is the encoded representation of the inputs
    input_layer = Input(shape=(input_dim, )) 
    encoded = Dense(hidden1_dim, activation="relu")(input_layer) 
    encoded = Dense(hidden2_dim, activation="relu")(encoded) 
    encoded = Dense(encoding_dim, activation="relu")(encoded) 

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(hidden2_dim, activation='relu')(encoded) 
    decoded = Dense(hidden1_dim, activation='relu')(decoded) 
    decoded = Dense(input_dim, activation='sigmoid')(decoded) 

    # this model maps an input to its reconstruction
    model = Model(inputs=input_layer, outputs=decoded)  
    if verbose_mode != 0:
        print("\n\n* model model summary")
        model.summary()

    # Separate Encoder model

    # this model maps an input to its encoded representation
    encoder = Model(input_layer, encoded)
    if verbose_mode != 0:
        print("\n\n* Encoder summary")
        encoder.summary()

    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim, ))

    # retrieve the layers of the autoencoder model
    decoder_layer1 = model.layers[-3]
    decoder_layer2 = model.layers[-2]
    decoder_layer3 = model.layers[-1]

    # create the decoder model
    decoder = Model(inputs=encoded_input, 
                    outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
    if verbose_mode != 0:
        print("\n\n* Decoder summary")
        decoder.summary()
    model.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['accuracy'])
    #model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])    
    
    return(model, encoder, decoder)

#******************************************************************************        
def fit_sae(model,
            x_train,
            x_test,
            y_train,
            y_test,
            checkpointer_file = "model.h5",
            tensorboard_logdir = './logs',
            nb_epoch = 5,
            batch_size = 32,
            verbose_mode = 1):
    # Let’s train our model for 10 epochs with a batch size of 32 samples and save the best performing model to a fil
    print("\n\n* Training the model...")
    if os.path.isfile(checkpointer_file):
        os.remove(checkpointer_file)
    checkpointer = ModelCheckpoint(filepath=checkpointer_file,
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir=tensorboard_logdir,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    start = time()
    history = model.fit(x_train, x_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        verbose=verbose_mode,
                        callbacks=[checkpointer, tensorboard]).history
    fit_time = time()-start
    if verbose_mode != 0:
        print("  Training time: ",secondsToStr(fit_time))
    return(history, fit_time)

#******************************************************************************        
def load_model_from(model_file):
    print("* Loading saved model...")
    model = load_model(model_file)
    
    return(model)

#******************************************************************************        
def eval_sae(model,
             x_train,
             x_test,
             eval_train = True,
             eval_test = True,
             batch_size = 32,
             verbose_mode = 1):
    # Evaluation
    start = time()
    if eval_train:
        print("*** Evaluating the train dataset...")
        val_train = model.evaluate(x_train, x_train, verbose=0)
    if eval_test:
        print("*** Evaluating the test dataset...")
        val_test = model.evaluate(x_test, x_test, verbose=0)
    if verbose_mode != 0:
        if eval_train or eval_test:
            print("         Accuracy       Loss")
        if eval_train:
            print("  Train: %8.3f  %8.7f" % (val_train[1], val_train[0]))
        if eval_test:
            print("  Test : %8.3f  %8.7f" % (val_test[1], val_test[0]))
        fit_time = time()-start
        print("  Evaluation time: ",secondsToStr(fit_time))
   
    if eval_train and eval_test:
        return(val_train[1], val_train[0], val_test[1], val_test[0])
    elif eval_train:
        return(val_train[1], val_train[0])
    elif eval_test:
        return(val_test[1], val_test[0])
    else:
        return
#******************************************************************************        
def plot_loss(hist,                       
              saveto_file):
    # model loss
    plt.title('model loss')
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')  
    if saveto_file != "":
        if os.path.isfile(saveto_file):
                os.remove(saveto_file)
        plt.savefig(saveto_file);   
    plt.close()
    
    return

#******************************************************************************        
def plot_accuracy(hist,                       
                  saveto_file):
    # model accuracy
    plt.title('model accuracy')
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')  
    if saveto_file != "":
        if os.path.isfile(saveto_file):
                os.remove(saveto_file)
        plt.savefig(saveto_file);         
    plt.close()
