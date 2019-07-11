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
import cnn2 
print('\n\n**************************************************************************')
print('\n\nFri1')
cnn2.main(in_path='C:\\IDSwDL\\', out_path='C:\\IDSwDL\\')
get_ipython().magic('reset -f')
