import os
import time
import pandas as pd
import yfinance as yf

def get_stock_info():
    """
    This function outputs all csv files to /assets/historical-symbols/ directory as individual CSV files per stocks
    """
    symbol_list = pd.read_csv('assets/symbols.csv')['Symbols'].tolist()
    for ticker in symbol_list:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='max')
            download_df = yf.download(ticker, period='max', progress=False)
            # rename everything ex: Open from download is now Open_adj
            download_df = download_df[['Open', 'High', 'Low', 'Close', 'Adj Close']]
            hist = hist.merge(download_df, left_index=True, right_index=True,  how='outer', suffixes=('', '_adj'))
            hist['sector'] = stock.info['sector']
            hist['ticker'] = ticker
            csv_name = 'assets/historical-symbols/' + ticker + '.csv'
            hist.to_csv(csv_name)
            print("Saved file for ", ticker)
            time.sleep(1.5)
        except KeyError:
            print('Failed on file ', ticker)
            continue
    print("Done!")


if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'assets/historical-symbols')
    if not os.path.exists(path):
        os.mkdir(path)
    get_stock_info()


