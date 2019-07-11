# -*- coding: utf-8 -*-
"""
created on sat may 25 01:33:52 2019

@author: b.baligh@gmail.com
"""

# %% imports
import pandas as pd
import glob
import os


# %% analyzer & statistics 
encoding = 'utf-8'

"""
*******************************************************************************
my_read_csv: 
*******************************************************************************
"""
def my_read_csv(file_name):
    missing_values = ["n/a", "na", "--"]
    DataF = pd.read_csv(file_name, 
                        encoding=encoding, 
                        na_values = missing_values,
                        skipinitialspace=True)
    return(DataF)

"""
*******************************************************************************
expand_categories: expands categories
*******************************************************************************
"""
def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v,round(100*(s[v]/t),2)))
    return "[{}]".format(",".join(result))

"""
*******************************************************************************
anlyze: analyze one csv file (filename)
*******************************************************************************
"""
def analyze(filename):
    print("* analyzing dataset file")
    print("dataset filename: {}".format(filename))
    
    df = pd.read_csv(filename,encoding=encoding)
    cols = df.columns.values
    total = float(len(df))
    
    print("{} rows".format(int(total)))
    print("{} cols".format(len(cols)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

    return(df) 

"""
*******************************************************************************
anlyze_all: analyze all csv files in pathname
*******************************************************************************
"""
def analyze_all(pathname):
    print("* analyzing dataset files")

    df = pd.concat(map(my_read_csv, glob.glob(os.path.join(pathname, "*.csv"))))
    
    cols = df.columns.values
    total = float(len(df))
    
    print("{} rows".format(int(total)))
    print("{} cols".format(len(cols)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count>100:
            print("** {}:{} ({}%)".format(col,unique_count,int(((unique_count)/total)*100)))
        else:
            print("** {}:{}".format(col,expand_categories(df[col])))
            expand_categories(df[col])

    return(df) 
    
"""
*******************************************************************************
load_all: load all csv files in pathname
*******************************************************************************
"""
def load_all(pathname,
             wildcard='*.csv'):
    print("* loading dataset files")

    df = pd.concat(map(my_read_csv, glob.glob(os.path.join(pathname, wildcard))))
    #df = pd.read_csv('c:\\idswdl\\dataset\\wednesday-workinghours.pcap_iscx.csv',
    #                 sep='\s*,\s*', header=0, engine='python',encoding=encoding)
    nb = len(glob.glob(os.path.join(pathname, "*.csv")))
    cols = len(df.columns.values)
    rows = int(float(len(df)))
    
    return(df, cols, rows, nb) 

"""
*******************************************************************************
get_next_batch: split the dataframe into smaller dataframes contained in a list
*******************************************************************************
"""
def get_next_batch(frame, indx, chunk_size):
    n = chunk_size  #chunk row size
    frame_chunk = frame[indx*n:(indx+1)*n]
    return(frame_chunk)