import os
import time
import pandas as pd
import numpy as np
import datetime as dt
import glob

def daily_features():
    
    path = r'assets/historical-symbols' # use your path
    all_files = glob.glob(path + "/*.csv")

    # Creating list to append all ticker dfs to
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    # Concat all ticker dfs
    stock_df = pd.concat(li, axis=0, ignore_index=True,sort=True)

    stock_df['Date'] = pd.to_datetime(stock_df['Date'])

    # Creating Moving Average Technical Indicator
    # Using this aritcle https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for quantitative-trading-8d98751b5fb
    stock_df['SMA_5'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window = 5).mean())
    stock_df['SMA_15'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window = 15).mean())
    stock_df['SMA_ratio'] = stock_df['SMA_15'] / stock_df['SMA_5']

    # Bollinger bands
    stock_df['SD'] = stock_df.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=15).std())
    stock_df['upperband'] = stock_df['SMA_15'] + 2*stock_df['SD']
    stock_df['lowerband'] = stock_df['SMA_15'] - 2*stock_df['SD']
    
    # Creating datetime date and making index
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df.index = stock_df['Date']
    stock_df.drop('Date',axis='columns',inplace=True)
    
    return stock_df

def quarterly_features():
    
    
    return None
