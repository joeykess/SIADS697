import os
import time

import yfinance as yf

DJI_stocks = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM',
              'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']


def get_stock_info(symbol_list):
    """
    This function outputs all csv files to /assets/ directory as individual CSV files per stocks
    :param symbol_list: list of stock symbols to get historical data on
    """
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
    path = os.path.join(cwd, 'assets')
    if not os.path.exists(path):
        os.mkdir(path)
    get_stock_info(DJI_stocks)
