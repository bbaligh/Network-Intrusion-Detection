# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:03:12 2019

@author: b.baligh@gmail.com
"""
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from scipy import interp

#from itertools import cycle
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

#from scipy.interpolate import spline

from time import time
from misc_func import secondsToStr
    
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
    if len(y_labels) == 1:
        y_labels = np.append(y_labels,["* SUSPECT *"])
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    
    y = labels.reshape(1,len(labels))
    y = y[0]
    y = le.transform(y)
    y = to_categorical(y)

    if verbose_mode != 0:
        print("Labels: ", y_labels)
    
    return(y, y_labels)

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
    if n_outputs == 1:
        n_outputs = 2
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
    
#******************************************************************************        
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
    if plot_loss:
        if plot_acc:
            plt.subplot(211)
        plt.title('Loss')
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='test')
        plt.legend()
    
    # plot accuracy during training
    if plot_acc:
        if plot_loss:
            plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(hist.history['acc'], label='train')
        plt.plot(hist.history['val_acc'], label='test')
        plt.legend()
    #plt.show()
    if (plot_acc or plot_loss) and (saveto_file != ""):
        if os.path.isfile(saveto_file):
                os.remove(saveto_file)
        plt.savefig(saveto_file)
    return

#******************************************************************************
def plot_roc_auc(y_test,
                 y_score,
                 labels,
                 saveto_file=None):
    
    #roc = roc_auc_score(y_test, y_score)
    #print("roc auc score = {}".format(roc))

    n_classes = y_test.shape[1]
    
    lw = 2
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC AUC')
    plt.legend(loc="lower right")
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    #plt.axis('equal')
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    if saveto_file != "":
        if os.path.isfile(saveto_file):
                os.remove(saveto_file)
        plt.savefig(saveto_file)
    plt.close()
    """
    import scikitplot as skplt
    skplt.metrics.plot_roc(y_test, y_score)
    plt.show()
    """
    return

#******************************************************************************
def plot_confusion_matrix(y_test, 
                          y_score, 
                          labels,
                          normalize=False,
                          mixed=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          saveto_file=None):
    
    np.set_printoptions(precision=4)
    y_score = np.argmax(y_score, axis=1)
    y_test = np.argmax(y_test, axis=1)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if mixed:
        normalize = False
        if not title:
            title = 'Mixed confusion matrix'
    else:
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_score)
    # Only use the labels that appear in the data
    labels = labels[unique_labels(y_test, y_score)]
    """
    Mixed mode confusion matrix
    """
    if mixed:
        """
        Confusion matrix without normalization
        """
        print('Confusion matrix, without normalization')
        print(cm)
        
        fig = plt.figure()
        st = fig.suptitle(title, fontsize="x-large")
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(121)
        #fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.grid(False)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title='Confusion matrix, without normalization',
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        
        """
        Normalized confusion matrix
        """
        cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print(cm)
        ax = fig.add_subplot(122)
        #fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.grid(False)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title='Normalized confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        
        st.set_y(0.95)
        fig.subplots_adjust(top=0.85)

        if saveto_file != "":
            if os.path.isfile(saveto_file):
                os.remove(saveto_file)
            plt.savefig(saveto_file)
        #plt.close()
        plt.show()
    else:
        if normalize:
            cm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.grid(False)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
    
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        if saveto_file != "":
            if os.path.isfile(saveto_file):
                os.remove(saveto_file)
            plt.savefig(saveto_file)
        #plt.close()
        plt.show()
    return