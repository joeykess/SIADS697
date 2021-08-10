#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def sharpe_ratio(dailys, rf=0.0015):
    """
    :param dailys: a list or array of daily returns
    :param rf: a float the risk free rate assumes 0.15% based on current conditions
    """
    ex = np.mean(dailys)*365
    sig = np.std(dailys, dtype=np.float64)*np.sqrt(365)
    sharpe = (ex - rf)/sig
    return sharpe

def sortino_ratio(dailys, mar=0.00):
    """
    :param dailys: a list or array of daily returns
    :param mar: minimum accepted return
    """
    mar_adj = [i for i in dailys if i<mar]
    sem_sig = np.std(mar_adj, dtype=np.float64)*np.sqrt(365)
    ex = np.mean(dailys)*365
    sort = (ex - mar)/sem_sig
    return sort

def treynor(dailys, BM, rf=0.0015):
    """
    :param dailys: a list or array of daily returns
    :param BM: a list or array of daily returns of a benchmark
    :param rf: a float the risk free rate assumes 0.15% based on current conditions
    """
    ex = np.mean(dailys)*365
    b = np.array(dailys)
    a = np.array(BM)
    reg = np.polyfit(a, b,1)
    beta = reg[0]
    trey = (ex - rf)/beta
    
    return trey

def max_drawdow(dailys):
    """
    :param dailys: a list or array of daily returns
    """
    cum_ret = np.cumprod([1+i for i in dailys])
    local = pd.Series(cum_ret).expanding(min_periods=1).max()
    max_dd = ((cum_ret/local)-1).min()
    
    return max_dd

def calmer(dailys):
    """
    :param dailys: a list or array of daily returns
    """
    ex = np.mean(dailys)*365
    dd = max_drawdow(dailys)
    calm = ex/-dd
    return calm


# In[ ]:




