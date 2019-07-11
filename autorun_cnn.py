# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 22:44:59 2019

@author: hp
"""
from IPython import get_ipython
    
def clear_console():
    try:
        #get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass

    return

#******************************************************************************
# Friday: 1st file of the day
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nFri1')
cnn.main(in_path='C:\\IDSwDL\\Results\\Fri1\\', out_path='C:\\IDSwDL\\Results\\Fri1\\')
get_ipython().magic('reset -f')


#******************************************************************************
# Friday: All files (3) of the day
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nFri')
cnn.main(in_path='C:\\IDSwDL\\Results\\Fri\\', out_path='C:\\IDSwDL\\Results\\Fri\\')
get_ipython().magic('reset -f')


#******************************************************************************
# Monday: 1 file 
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nMon')
cnn.main(in_path='C:\\IDSwDL\\Results\\Mon\\', out_path='C:\\IDSwDL\\Results\\Mon\\')
get_ipython().magic('reset -f')

#******************************************************************************
# Tuesday: 1 file
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nTue')
cnn.main(in_path='C:\\IDSwDL\\Results\\Tue\\', out_path='C:\\IDSwDL\\Results\\Tue\\')
get_ipython().magic('reset -f')


#******************************************************************************
# Wednesday: 1 file
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nWed')
cnn.main(in_path='C:\\IDSwDL\\Results\\Wed\\', out_path='C:\\IDSwDL\\Results\\Wed\\')
get_ipython().magic('reset -f')


#******************************************************************************
# Thursday: All files (2) of the day
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nThu')
cnn.main(in_path='C:\\IDSwDL\\Results\\Thu\\', out_path='C:\\IDSwDL\\Results\\Thu\\')
get_ipython().magic('reset -f')


#******************************************************************************
# All dataset: 8 files
#******************************************************************************
import cnn 
print('\n\n**************************************************************************')
print('\n\nAll')
cnn.main(in_path='C:\\IDSwDL\\Results\\All\\', out_path='C:\\IDSwDL\\Results\\All\\')
get_ipython().magic('reset -f')

