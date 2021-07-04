import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json


def candle_charts(ticker, start, end):
    """
    Takes in ticker, start, and end dates and generates candle charts with EMA15 and upper and lower Bollinger Bands,
    then saves file at assets/cnn_files/{ticker}{start}_{end}.png
    :param ticker: standard stock ticker (BA, AAPL, GOOG, etc.)
    :param start: start date in format (yyyy-mm-dd -> 2019-01-01)
    :param end: end date in format (yyyy-mm-dd -> 2019-01-01)
    """
    df2 = pd.read_csv(f'assets/historical-symbols/{ticker}.csv')
    df2['MA15'] = df2['Close'].rolling(window=15).mean()

    def bollinger_bands(data, sma, window):
        std = data.rolling(window=window).std()
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        return upper_bb, lower_bb

    df2['upper_bb'], df2['lower_bb'] = bollinger_bands(df2['Close'], df2['MA15'], 15)
    df2['ema15'] = df2['Close'].ewm(span=15).mean()
    df = df2.loc[(df2['Date'] >= start) & (df2['Date'] <= end)].copy()
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    df = df.set_index(['Date'])
    df.index = df.index.to_pydatetime()
    fig, ax = plt.subplots(figsize=(18, 9))
    mpf.plot(df, type='candlestick', style='charles', ax=ax)
    df.reset_index().plot(kind='line', y='upper_bb', color='blue', lw=3, ax=ax, legend=None)
    df.reset_index().plot(kind='line', y='lower_bb', color='orange', lw=3, ax=ax, legend=None)
    df.reset_index().plot(kind='line', y='ema15', color='black', lw=3, ax=ax, legend=None)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.axis('off')
    if not os.path.exists('assets/cnn_images/{}'.format(ticker)):
        os.makedirs('assets/cnn_images/{}'.format(ticker))
    plt.savefig('assets/cnn_images/{}/{}_{}.png'.format(ticker, start, end), dpi=300, pad_inches=0.05)
    plt.close('all')


def make_charts(ticker, num_days):
    dict_of_results = {}
    data2 = pd.read_csv('assets/historical-symbols/{}.csv'.format(ticker))
    data = data2.loc[(data2['Date'] <= '2020-01-01') & (data2['Date'] >= '2010-01-01')].copy()
    for start in range(0, data.shape[0], num_days):
        df_subset = data.iloc[start:start + num_days].copy()
        curr_max = df_subset['Close'].max()
        next_max = data.iloc[start + num_days:start + 2 * num_days]['Close'].max()
        beg = df_subset['Date'].iloc[0]
        end = df_subset['Date'].iloc[-1]
        if curr_max == next_max:
            val = 0
        elif curr_max < next_max:
            val = 1
        else:
            val = -1
        dict_of_results['{}_{}.png'.format(beg, end)] = val
        candle_charts(ticker, beg, end)
    a_file = open("{}.json".format(ticker), 'w')
    json.dump(dict_of_results, a_file)
    a_file.close()


if __name__ == '__main__':
    # testing candle_charts functionality
    # candle_charts('BA', '2015-01-01', '2015-01-22')
    symbols_list = pd.read_csv('symbols.csv')['Symbols'].tolist()
    for symbol in tqdm(symbols_list):
        try:
            make_charts(symbol, 10)
        except Exception as e:
            print(e)
        break
