import os
import time
import pandas as pd
import yfinance as yf


def get_stock_info():
    """
    This function outputs all csv files to /assets/ directory as individual CSV files per stocks
    :param symbol_list: list of stock symbols to get historical data on
    """
    symbol_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    for ticker in symbol_list:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='max')
        hist['sector'] = stock.info['sector']
        hist['ticker'] = ticker
        csv_name = 'assets/' + ticker + '.csv'
        hist.to_csv(csv_name)
        print("Saved file for ", ticker)
        time.sleep(5)
    print("Done!")


if __name__ == '__main__':
    cwd = os.getcwd()
    # check if OS is  ('nt') or not
    if os.name == 'nt':
        path = os.path.join(cwd, '/assets')
    else:
        path = os.path.join(cwd, 'assets')
    if not os.path.exists(path):
        os.mkdir(path)
    get_stock_info()

