import os
import time
import pandas as pd
import requests
import csv
from tqdm import tqdm


def get_short_stock_info():
    """
    This function outputs all csv files to /assets/historical-symbols/ directory as individual CSV files per stocks
    Will be choosing 10 stocks that make up the top of the S&P500
    """
    symbol_list = ['NVDA', 'AMD', 'JPM', 'JNJ', 'MRNA', 'F', 'TSLA', 'MSFT', 'BAC', 'BABA', 'SPY', 'QQQ']
    # ones that work with alpha vantage (no nasdaq)
    # chosen for high volume and availability on alpha vantage
    av_api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    slices = ['year{}month{}'.format(a, b) for a in range(1, 3) for b in range(1, 13)]
    for ticker in tqdm(symbol_list):
        ticker_df = pd.DataFrame()
        for slice in slices:
            # each request takes approximately 3 seconds
            csv_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=' \
                      '{}&interval=5min&slice={}&apikey={}'.format(ticker, slice, av_api_key)
            new_df = pd.read_csv(csv_url, header=0)
            ticker_df = ticker_df.append(new_df)
        ticker_df.to_csv('assets/short_term_symbols/{}.csv'.format(ticker))
        # sleeps 15 seconds just to make sure no timeout is incurred
        time.sleep(15)
    print("Done!")


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'assets/short_term_symbols')
    if not os.path.exists(path):
        os.mkdir(path)
    get_short_stock_info()
