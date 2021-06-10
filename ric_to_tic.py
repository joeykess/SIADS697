#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import re


# In[12]:


def ric_to_sym(fund_file):
    '''takes in the fundamentals data csv extracts RICS, and converts them to common ticker sybmols for continuity'''
    df = pd.read_csv(fund_file, index_col=0)
    inst_ric = df['Instrument'].unique()
    tics = [i.split('.')[0] for i in inst_ric]
    tics =  ['BRK-B' if i == 'BRKb' else i for i in tics]
    tics =  ['BF-B' if i == 'BFb' else i for i in tics]
    symbols = pd.DataFrame({'Symbols':tics})
    symbols.to_csv('symbols.csv', index = False)


# In[ ]:


if __name__ == '__main__':
    fund_file = input('Path?')
    ric_to_sym(fund_file)

