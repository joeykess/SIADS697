import os
import time
import pandas as pd
import yfinance as yf

def get_recommendation_info():
    """
    This function outputs a single csv file to /assets directory with recommendations for our entire ticker list
    """
    
    symbol_list = pd.read_csv('assets/symbols.csv')['Symbols'].tolist()
    rec_df = pd.DataFrame(columns=['Date','Firm','To Grade','Action','ticker'])
    failed_ticker_list = []
    
    for ticker in symbol_list:
        
        try:
            stock = yf.Ticker(ticker)
            reco = stock.recommendations

            try:
                reco['ticker'] = ticker
                rec_df = rec_df.append(reco)
                print(f'Loaded {ticker}')
                
            except TypeError:
                print(f'No Recommendations for {ticker}')
                pass
            
            time.sleep(1.5)

        except KeyError:
            print(f'Failed to load {ticker}')
            failed_ticker_list.append(ticker)
            continue

    rec_df.to_csv('assets/ticker_recommendations.csv')
        
        
if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path.join(cwd, 'assets')
    if not os.path.exists(path):
        os.mkdir(path)
    get_recommendation_info()