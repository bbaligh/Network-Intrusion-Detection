# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 22:54:09 2019

@author: b.baligh@gmail.com
"""
from time import strftime, localtime, time
from datetime import timedelta

#******************************************************************************
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))
