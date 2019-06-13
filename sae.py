# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:56:06 2019

@author: b.baligh@gmail.com
"""
# %% Imports

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import os
#import glob
#import shutil
#import math
import ds_tools
import pandas as pd 
import pickle 
from scipy import stats 
import seaborn as sns 
from pylab import rcParams 
from keras.models import Model, load_model 
from keras.layers import Input, Dense 
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from keras.layers import LeakyReLU

#%matplotlib inline 

#%% 
#Global variables
dataset_path = "C:\\IDSwDL\\Dataset"

# %% 
#Display dataset file Stats 

#dataset_filename = "c:/IDSwDL/Dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
#ds_tools.analyze(dataset_filename)
#files = [f for f in glob.glob(dataset_path + "**/*.csv", recursive=False)]
#i = 1
#for f in files:
#    print()
#    print("Dataset file N°{}".format(i))
#    ds_tools.analyze(f)
#    i = i+1
#print("Total dateset files: {}".format(len(files)))
#

# %%
# Display all dataset files Stats located in dataset_path
def display_stats():
    ds_tools.analyzeAll(dataset_path)
    return

# %%
# Load & Display minimum info about all dataset files Stats located in dataset_path
def load_ds(lVerbose):
    df, cols, rows, _ = ds_tools.loadAll(dataset_path)
    if lVerbose:
        print("{} rows".format(rows))
        print("{} cols".format(cols))
        
    return(df, cols, rows)


# %% Main
if __name__ == '__main__':
    
    sns.set(style='whitegrid', palette='muted', font_scale=1.5) 
    rcParams['figure.figsize'] = 14, 8 
    RANDOM_SEED = 42 
    LABELS = []
    
# %% 
    df, c, _ = load_ds(True)
    
    print("df:",df.shape)
    
# %% df
    LABELS = df['Label'].unique()
    print("* Counting Label class frequency and distribution...")
    count_classes = pd.value_counts(df['Label'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.title("'Label' class distribution")
    #plt.xticks(range(5), LABELS)
    plt.xlabel("'Label' class")
    plt.ylabel("Frequency")
    plt.savefig('label_class_distribution.png')
# %% 
    attacks = df[df.Label != "BENIGN"] 
    normal = df[df.Label == "BENIGN"] 
   
    print("attacks:",attacks.shape)
    print("normal:",normal.shape)
    
# %%
    # Checking for Null values
    print("* Checking for null values")
    print("  Null values: {}".format(df.isnull().values.any()))    
    print("  # of Null values: {}".format(df.isnull().sum().sum())) 
    
# %%    
    # Checking for inf values
    print("* Checking for inf values")
    df[df == np.inf] = np.nan
    df[df == -np.inf] = np.nan
    #df.replace([np.inf, -np.inf], np.nan)
    print(df.isnull().sum())
    df = df.replace(r'^\s+$', np.nan, regex=True)
    print(df.isnull().sum())
    
# %%    
    # Replacing nan values by mean()
    print("* Replacing nan values by mean values") 
    for i in range(df.shape[1]-1):
        mean_value = df[df.columns[i]].mean()
        print("\rReplacing nan values in column {} with {}".format(i, mean_value), end="")
        df[df.columns[i]]=df[df.columns[i]].fillna(mean_value)
    #df.fillna(df.mean(), inplace=True)
    
# %% 
    # Preparing the data
    print("* Normalizing the data...")
    for i in range(df.shape[1]-1):
        print("\rReshaping column: {}".format(i), end="")
        #df[df.columns[i]] = StandardScaler().fit_transform(df[df.columns[i]].values.reshape(-1, 1))
        df[df.columns[i]] = MinMaxScaler().fit_transform(df[df.columns[i]].values.reshape(-1, 1))
        
# %% 
    print("* Preparing the data...")
    # Y = last column contains labels
    Y = df.iloc[:,c-1].values 
    # X = all of the dataframe except the last column which is the labels column
    X = df.iloc[:,0:c-1].values
 

# %% 
    # split dataframe to train and test 
        # Preparing the data
    print("* Splitting the data (train & test)...")
    #train, test = train_test_split(df, col_true, stratify=col_true, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)

# %%    
    # normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
    x_train = (x_train.astype('float32')+1) / 2.
    x_test = (x_test.astype('float32')+1) / 2.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# %%
    print("="*50)
    print(x_train.shape[0])
    print(x_test.shape[0])

# %%    
    # binary classification
    
    y_train[y_train == "BENIGN"] = 0
    y_train[y_train != 0] = 1
    y_test[y_test == "BENIGN"] = 0
    y_test[y_test != 0] = 1
    
    # n'ary classifcation
    """
    y_labels = pd.DataFrame(Y)[0].unique()
    print("Labels: ",y_labels)
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    """
# %%%    
    # convert values from object to float32
    y_train=y_train.astype('float32')
    y_test=y_test.astype('float32')

    
# %% 
    """
    x_train, x_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    x_train = x_train[x_train.Label == 'BENIGN']
    x_train = x_train.drop(['Label'], axis=1)
    y_test = x_test['Label']
    x_test = x_test.drop(['Label'], axis=1)
    x_train = x_train.values
    x_test = x_test.values
    x_train.shape   
    """
# %% 
    # Building the model for the SAE
    print("Building the model...")
    input_dim = x_train.shape[1] #78
    hidden1_dim = 64
    hidden2_dim = 50
    encoding_dim = 25 # 5x5 for the next CNN network
    
    compression_factor = float(input_dim) / encoding_dim
    print("Compression factor: %s" % compression_factor)

# %%         
    # "encoded" is the encoded representation of the inputs
    input_layer = Input(shape=(input_dim, )) 
    #encoded = Dense(input_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer) 
    #encoded = LeakyReLU(alpha=0.3)(encoded)
    encoded = Dense(hidden1_dim, activation="relu")(input_layer) 
    encoded = Dense(hidden2_dim, activation="relu")(encoded) 
    encoded = Dense(encoding_dim, activation="relu")(encoded) 
# %%    
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(hidden2_dim, activation='relu')(encoded) 
    decoded = Dense(hidden1_dim, activation='relu')(decoded) 
    decoded = Dense(input_dim, activation='sigmoid')(decoded) 
# %%    
    # this model maps an input to its reconstruction
    SAE = Model(inputs=input_layer, outputs=decoded)  
    SAE.summary()
# %%    
    # Separate Encoder model

    # this model maps an input to its encoded representation
    encoder = Model(input_layer, encoded)
    encoder.summary()
# %%    
    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim, ))
# %%
    # retrieve the layers of the autoencoder model
    decoder_layer1 = SAE.layers[-3]
    decoder_layer2 = SAE.layers[-2]
    decoder_layer3 = SAE.layers[-1]
# %%    
    # create the decoder model
    decoder = Model(inputs=encoded_input, 
                    outputs=decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
    decoder.summary()
    
# %% 
    # Let’s train our model for 100 epochs with a batch size of 32 samples and save the best performing model to a fil
    print("Training the model...")
    nb_epoch = 10
    batch_size = 32 #32
    SAE.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #SAE.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
# %%    
    checkpointer = ModelCheckpoint(filepath="model.h5",
                                   verbose=0,
                                   save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
# %%
    history = SAE.fit(x_train, x_train,
                      epochs=nb_epoch,
                      batch_size=batch_size,
                      shuffle=True,
                      validation_data=(x_test, x_test),
                      verbose=1,
                      callbacks=[checkpointer, tensorboard]).history
# %% 
    # And load the saved model (just to check if it works):
    print("Loading the saved model...")
    SAE = load_model('model.h5')    
# %% 
    """
    predictions = SAE.predict(x_test)
    for ii in range(78):
        plt.title('predict var_{} ({})'.format(ii, df.columns[ii]))
        plt.scatter(x_test[:, ii], predictions[:, ii])
        plt.show()
        plt.savefig('predict_var_{}.png'.format(ii));
    """
# %%
    # Evaluation
    print("* Evaluating the model...")
    val = SAE.evaluate(x_train, x_train, verbose=0)
    print("  Train evaluation:\n  loss = {} \n  accuracy = {}".format(val[0], val[1]))
    val = SAE.evaluate(x_test, x_test, verbose=0)
    print("  Test evaluation:\n  loss = {} \n  accuracy = {}".format(val[0], val[1]))

# %%
    # model loss
    plt.title('model loss')
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')  
    plt.savefig('model_loss.png');    
    
# %%
    # model accuracy
    plt.title('model accuracy')
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')  
    plt.savefig('model_accuracy.png');         

# %%
    predictions = SAE.predict(x_test, verbose=1)
    mse = np.mean(np.power(x_test - predictions, 2), axis=1)
    error_df = pd.DataFrame({'reconstruction_error': mse,
                            'true_class': y_test})
    error_df.describe()
    
# %%
    # Reconstruction error without attacks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
    normal_error_df = error_df[(error_df['true_class']== 0)]
    _ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
    plt.savefig('error_without_attacks.png');
    
# %%
    # Reconstruction error with attacks
    fig = plt.figure()
    ax = fig.add_subplot(111)
    attacks_error_df = error_df[error_df['true_class'] != 0]
    _ = ax.hist(attacks_error_df.reconstruction_error.values, bins=10)
    plt.savefig('error_with_attacks.png');
    
# %%
    # ROC curves
    from sklearn.metrics import (confusion_matrix, 
                                 precision_recall_curve, 
                                 auc, 
                                 roc_curve, 
                                 recall_score, 
                                 classification_report, 
                                 f1_score, 
                                 precision_recall_fscore_support)
    
    fpr, tpr, thresholds = roc_curve(error_df.true_class, 
                                     error_df.reconstruction_error)
    roc_auc = auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig('ROC_curve.png');
    
# %%    
    # Precision vs Recall
    precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
    plt.plot(recall, precision, 'b', label='Precision-Recall curve')
    plt.title('Recall vs Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.show()
    plt.savefig('precsion_curve.png')
    
# %%    
    # Precision for different threshold values    
    plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
    plt.title('Precision for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    #plt.show()
    plt.savefig('precsion_threshold_values.png');
    
# %%    
    # Recall for different threshold values    
    plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
    plt.title('Recall for different threshold values')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Recall')
    #plt.show()
    plt.savefig('recall_threshold_values.png');
    
# %%
    # Prediction
    threshold = 2.9
    groups = error_df.groupby('true_class')
    fig, ax = plt.subplots()
    
    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Attack" if name == 1 else "Normal")
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction error for different classes")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    #plt.show()
    plt.savefig('prediction.png');
    
# %%
    # Confusion matrix
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    plt.figure(figsize=(12, 12))
    #sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    sns.heatmap(conf_matrix, 
                xticklabels=['Normal', 'Attacks'], 
                yticklabels=['Normal', 'Attacks'], 
                annot=True, 
                fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    #plt.show()
    plt.savefig('confusion_matrix.png');

# %%    
    # X encoding
    x_encoded = encoder.predict(X)
# %%
    # Saving x_encoded to a CSV file
    pd.DataFrame(x_encoded).to_csv('x_encoded.csv', index=False)
    pd.DataFrame(Y).to_csv('labeled.csv', index=False)
