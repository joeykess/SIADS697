#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import os
import re
import warnings
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None 


# In[2]:


def create_running_df():
    """
    :return: returns a dataframe which includes all stock values per day since earliest tracked day in yfinance
    formatted like: date | AAPL | BA | CSCO | MSFT...etc
    """
    try:
        df = pd.read_csv('assets/hist_dailies.csv')
    except:
        path_to_files = 'assets/historical-symbols'
        df = pd.DataFrame()
        for file in os.listdir(path_to_files):
            try:
                if file.endswith('.csv'):
                    symbol = re.findall(r'(.*).csv', file)[0]
                    temp_df = pd.read_csv(path_to_files + '/' + file).set_index(['Date'])
                    temp_df[symbol] = temp_df['Close']
                    # Removing averaging of price to use more accurate buy/sell price
#                     temp_df[symbol] = temp_df[['Close', 'Open', 'High', 'Low']].mean(axis=1)
                    # since we're trading in the 'middle' of the day so we can't assume pricing is Open or Close
                    if df.shape[0] == 0:
                        df = temp_df[symbol].to_frame(name=symbol)
                    else:
                        df = df.join(temp_df[symbol].to_frame(name=symbol))
            except Exception as e:
                print(e, file)
        df.to_csv('assets/hist_dailies.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index(['Date'])

# Helper function to allow user to create list of tickers sell (Note: All positions are sold)
def create_sell_dict(ticker_preds,open_positions_dict,sell_threshold = -0.05):

    # Establish weights for selling stocks - Not working, only selling full positions
    # How do we deal with cascading price reductions (e.g. only lose 5%, but time after time)
    sell_stocks = ticker_preds[ticker_preds['earn_ratio'] <= sell_threshold]

    sell_dict = {}

    for ticker in sell_stocks.ticker.unique():
        try: 
            # Sell all open positions
            sell_dict[ticker] = open_positions_dict[ticker]

        except: # or pass if they do not exist
            pass

    return sell_dict

# Helper function to allow user to create dictionary of tickers and quantities to buy
def create_buy_dict(ticker_preds,current_cash,buy_threshold=0.05,buy_ratios=None):

    # Establishing weights to buy stocks
    buy_stocks = ticker_preds[ticker_preds['earn_ratio'] >= buy_threshold].sort_values(by='earn_ratio',ascending=False)

    buy_dict = {}

    # Establish weights for buying stocks. Allows you to bring in your owns buy ratios
    if buy_ratios == None:
        buy_ratios = buy_stocks.earn_ratio / buy_stocks.earn_ratio.sum()
    else:
        pass

    # Getting cash amount that can be used to buy; Can only buy full shares
    buy_dict = dict(zip(buy_stocks.ticker, (current_cash * buy_ratios / buy_stocks.curr_price).astype('int')))

    # Removing keys where we are not buying any stock
    del_list = []
    for key in buy_dict.keys():
        if buy_dict[key] == 0:
            del_list.append(key)

    for key in del_list:
        del buy_dict[key]

    return buy_dict

# Helper function to allow user to get portfolio value for current open positions for given date
def get_curr_portfolio_value(open_positions_dict, port, date):

    output_dict = {}
    
    # Getting key closest to date picked, but in past
    try:
        open_positions_dict[date]
        date_key = date
    except:
        date_key = [x for x in open_positions_dict.keys() if x < date][-1]

    for ticker, quantity in open_positions_dict[date_key].items():

        curr_price = port.get_price(date, ticker)

        output_dict[ticker] = {'date':date,'price':curr_price,'quantity':quantity,'curr_val':curr_price*quantity}

    return output_dict


# In[3]:


class portfolio:
    def __init__(self, start_date='2015-01-05', value=1000000, end_date='2021-07-02'):
        """
        :param start_date: beginning date to start portfolio
        :param value: amount of initial cash to use to buy shares
        :param end_date: end date of trading simulation - 2021-07-02 is last day in data so that is hard coded for now
        :var: current_cash is the amount of cash held at any given moment in the account duration
        :var: open_positions_dict is a dictionary which at any given point has the held positions for instance {AAPL:10}
        :var: hist_trades_dict and df track the historical trades, where the dataframe keeps a column to track buy/sell
        :var: hist_cash_dict is for all of the cash changes - from buy/sell/add to account
        """
        self.end_date = end_date
        self.start_date = start_date
        self.init_cash = value
        self.current_cash = value
        self.track_record = pd.DataFrame(columns = ['Date', 'Value'])
        self.snapshots = {}
        self.open_positions_dict = {}
        self.open_positions_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity','Basis','Purchase Price', "Current Value", "Last", "% Gain"])
        self.tracking_df = create_running_df()
        self.tracking_df.index = pd.to_datetime(self.tracking_df.index)
        self.hist_trades_dict = {}
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])
        #self.hist_cash_df = pd.DataFrame([[start_date, value]], columns=['Date', 'Cash Available to Trade'])
        #self.hist_cash_dict = {datetime.strptime(start_date, '%Y-%m-%d'): value}
    def reset_account(self):
        self.__init__(self.start_date, self.init_cash, self.end_date)
    def add_cash(self, amount, date):
        """
        :param amount: amount of cash you want to add to account
        :param date: date you want to add money to account
        """
        # adds amount to current cash
        self.current_cash += amount
        # adds as cash order to trade tracker
        self.hist_trades_df = self.hist_trades_df.append(
            {'Date': date, 'Order Type': 'cash',
             'Ticker': 'NA', 'Quantity': 'NA',
             'Ticker Value': 'NA', 'Total Trade Value': amount,
             'Remaining Cash': self.current_cash}, ignore_index=True)
        # adds cash to historical cash dictionary
        self.hist_cash_dict[datetime.strptime(date, '%Y-%m-%d')] = self.current_cash
        
    def execute_trades(self, order, date, t_type):
        """
        :param order: dictonary that includes tickers as keys and shares as the items
        :param date: date of the trade in string format YYYY-MM-DD
        :param t_type: trade type (buy or sell)"""
        
        blotter = pd.DataFrame(order, index =[0]).T.reset_index().rename(columns = {'index': 'Ticker', 0: 'Quantity'}) ###creates a trade blotter
        blotter['Transaction'] = t_type.upper() ###uniform info
        trade_tickers = list(blotter['Ticker']) ###Stocks I am buying
        trade_shares = list(blotter['Quantity']) ###No of Shares
        port = self.open_positions_df ###my portfolio before trades
        holdings = list(port['Ticker']) ###stocks in my portfolio before trades
        px = [self.get_price(date, i) for i in trade_tickers] ####Prices for stocks I am buying
        dates = [date for i in trade_tickers] ###date of record for each trade
        blotter['Last'] = px
        if t_type.upper() == 'BUY': ###Run Buys
            if len(holdings) == 0: ###If the portfolio is empty
                df_to_positions = pd.DataFrame({'Date':dates, 'Ticker':trade_tickers, 'Quantity':trade_shares,'Purchase Price':px,"Last":px}) ###enter trades
                df_to_positions['Current Value'] = df_to_positions['Quantity'] * df_to_positions['Last'] ### establish the current value of a transaction
                df_to_positions['Basis'] = df_to_positions['Quantity'] * df_to_positions['Purchase Price'] ###establish the cost basis - this is the same as current value on day one for new stocks 
                df_to_positions['% Gain'] = round(((df_to_positions['Current Value']/df_to_positions['Basis'])-1),4)*100 ###calculates gain for each stock - will be 0% for new stocks
                df_to_positions = df_to_positions.filter(['Date', 'Ticker', 'Quantity','Basis','Purchase Price', "Current Value", "Last", "% Gain"])
                if df_to_positions['Basis'].sum()> self.current_cash: ####Ensures that you can execute your trade in its entirety.
                    print("You do not have Enough Cash to execute these trades, Please adjust and resubmit") ###If you dont have enough cash this will appear
                else:
                    print("Trades Executed")### if trades are validated this will appear
                    self.open_positions_df = df_to_positions ### if the portfolio is empty the new positions df becomes the open positions df
                    cash_impact = [] ### the loop accounts for the change in cash for each trade for the historical transactions blotter.
                    remaining = self.current_cash
                    for i in list(df_to_positions['Basis']):
                        x = remaining - i
                        remaining = x
                        cash_impact.append(x)
                    self.current_cash -= df_to_positions['Basis'].sum() ###cash is adjusted for trades
                    df_to_hist = df_to_positions.filter(['Date', 'Ticker', 'Quantity', 'Purchase Price', 'Basis']) 
                    df_to_hist['Order Type'] = t_type.upper() 
                    df_to_hist['Remaining Cash'] = cash_impact
                    df_to_hist = df_to_hist.rename(columns={'Purchase Price': 'Ticker Value', 'Basis':'Total Trade Value'})
                    df_to_hist.filter(['Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'])
                    self.hist_trades_df = df_to_hist #transaction history is updated
                    
            else: ###If the portfolio is not empty 
                df_to_positions = pd.DataFrame({'Date':dates, 'Ticker':trade_tickers, 'Quantity':trade_shares,'Purchase Price':px,"Last":px}) ###enter trades
                df_to_positions['Current Value'] = df_to_positions['Quantity'] * df_to_positions['Last'] ###establish value of a trade
                df_to_positions['Basis'] = df_to_positions['Quantity'] * df_to_positions['Purchase Price'] ###establish basis of the new positions
                df_to_positions['% Gain'] = round(((df_to_positions['Current Value']/df_to_positions['Basis'])-1),4)*100 ###gain here should also be 0
                df_to_positions = df_to_positions.filter(['Date', 'Ticker', 'Quantity','Basis','Purchase Price', "Current Value", "Last", "% Gain"]) ###set up trade df
                if df_to_positions['Basis'].sum()> self.current_cash: ###Ensures you can execute                    
                    print("You do not have Enough Cash to execute these trades, Please adjust and resubmit")
                else:
                    print("Trades Executed")                
                    self.open_positions_df = pd.concat([self.open_positions_df, df_to_positions]) ###Executes trades and add to EXISTING Portfolio
                    cash_impact = [] ###loop accounts for change in cash for transactions
                    remaining = self.current_cash
                    for i in list(df_to_positions['Basis']):
                        x = remaining - i
                        remaining = x
                        cash_impact.append(x)
                    self.current_cash -= df_to_positions['Basis'].sum() ###cash is adjusted for trades
                    df_to_hist = df_to_positions.filter(['Date', 'Ticker', 'Quantity', 'Purchase Price', 'Basis'])
                    df_to_hist['Order Type'] = t_type.upper()
                    df_to_hist['Remaining Cash'] = cash_impact
                    df_to_hist = df_to_hist.rename(columns={'Purchase Price': 'Ticker Value', 'Basis':'Total Trade Value'})
                    df_to_hist.filter(['Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'])
                    self.hist_trades_df = pd.concat([self.hist_trades_df, df_to_hist])###History is updated
            ### consolidate lots 
            consolidated = pd.DataFrame() ###will store consolidated data for each trade
            self.open_positions_df= self.open_positions_df.reset_index(drop = True)###reset index
            for stock in list(self.open_positions_df['Ticker'].unique()): ### grabs a stock from the portfolio
                lots = self.open_positions_df[self.open_positions_df['Ticker']==stock] ### Isolates all rows in the portfolio with that stock
                if len(lots)==1: ### Checks if there is only one row
                    consolidated = pd.concat([consolidated, lots]) ### if there is only one row of data for that stock add to the consolidate df
                else: ### if there are more than one
                    dt = lots['Date'].min()### the first trade date
                    q = lots['Quantity'].sum()### the total number of shares that I now have
                    basis = lots['Basis'].sum()### my new Basis for the stock
                    ppx = basis/q ### my average price 
                    cv = ppx * lots['Last'].iloc[-1] ### my current value 
                    gain = round((cv/basis)-1,4)*100 ### my adjusted gain
                    cons_lots = pd.DataFrame({'Date':dt, 'Ticker':stock, 'Quantity':q,'Basis':basis,
                                              'Purchase Price':ppx, "Current Value":cv, "Last":lots['Last'].iloc[-1], "% Gain":gain}, index = [0])### set the afore mentioned data up in a DF
                    consolidated = pd.concat([consolidated, cons_lots]) ###add the consolidated row to the consolidated df
            self.open_positions_df = consolidated ###overwrite the portfolio as the consolidated df
        else:
            port = self.open_positions_df ### grabs current portfolio
            stocks_in_port = list(port['Ticker']) ### grabs stocks in portfolio
            stocks_to_sell = list(blotter['Ticker']) ### grabs stocks in trade blotter
            for stock in stocks_to_sell: ### loop checks if stocks not in portfolio 
                if stock not in stocks_in_port:
                    print("Trade Rejected - Short Sell not allowd") ### if the stock is not in the portfolio the trade is rejected
                elif port[port['Ticker']==stock]['Quantity'].iloc[0]< blotter[blotter['Ticker']==stock]['Quantity'].iloc[0]:
                    print("Trade Rejected - Short Sell not allowd") ### if selling more than you own trade is rejected
                else:
                    trade_request = blotter[blotter['Ticker']==stock]### isolate the trade in the blotter
                    q_to_sell = trade_request['Quantity'].iloc[0] ### ammount selling
                    sell_px = trade_request['Last'].iloc[0] ### Last price for the stock
                    transaction = q_to_sell * sell_px ### Trade Value
                    self.current_cash+=transaction ### add value of the trade back to cash
                    current_pos = port[port['Ticker']==stock] ### isolate this stock in the portfolio
                    current_pos['Quantity'] = current_pos['Quantity'].iloc[0]-q_to_sell ### reduce the quantity owned
                    current_pos['Current Value'] = current_pos['Current Value'].iloc[0] - transaction ### Reduce the value in the portfolio
                    current_pos['Basis'] = current_pos['Purchase Price'] * current_pos['Quantity'] ### recalculate basis using new quantity
                    current_pos['Last'] = sell_px ### last price
                    current_pos["% Gain"] = round((current_pos['Current Value']/current_pos['Basis'])-1, 4) *100 ###calculate % gain in stock after adjustment.
                    self.open_positions_df = self.open_positions_df[self.open_positions_df['Ticker']!=stock] ### remove old line from portfolio
                    self.open_positions_df = pd.concat([self.open_positions_df, current_pos]) ### add new line into portfolio
        self.open_positions_df = self.open_positions_df[self.open_positions_df['Quantity']!=0] ### remove any lines where quantity is 0 (just in case)
        update_px = [self.get_price(date, i) for i in  self.open_positions_df['Ticker']] ###get updated prices for each stock
        self.open_positions_df['Last'] = update_px ###update price columns
        self.open_positions_df['Current Value'] = self.open_positions_df['Last'] * self.open_positions_df['Quantity'] ### update current value 
        self.open_positions_df["% Gain"] = round(((self.open_positions_df['Current Value']/self.open_positions_df['Basis'])-1),4)*100 ### update gain
        self.open_positions_df= self.open_positions_df.reset_index(drop = True)###clean up the index
        self.snapshots['Positions_{}'.format(date)] = self.open_positions_df.copy() ### update snapshot
        self.snapshots['cash_{}'.format(date)] = self.current_cash.copy() ### update snapshot
        val = self.open_positions_df['Current Value'].sum() + self.current_cash ### calculate portfolio value
        val_ex = self.open_positions_df['Current Value'].sum() ### calculate portfolio value EX cash
        self.snapshots['val_{}'.format(date)] = val # update Snapshot
        tr = pd.DataFrame({'Date':date, 'Value':val, 'Val_ex_cash': val_ex}, index = [0]) # update trackrecord
        self.track_record = pd.concat([self.track_record, tr]).reset_index(drop = True) # update trackrecord
        return 


    def get_price(self, date, ticker):
        '''
        :param date: date of price needed in string format YYYY-MM-DD
        :param ticker: stock ticker for price needed
        '''
        ticker = ticker.upper()
        conv_date = datetime.strptime(date, '%Y-%m-%d')
        return self.tracking_df.loc[conv_date][ticker]
    
    def update_portfolio(self, date, cash_add = None):
        '''
        :param date: date of price needed in string format YYYY-MM-DD
        :param cash_add: adds or subtracts cash from the portfolio on the date specified    
        '''
        px = [self.get_price(date, t) for t in list(self.open_positions_df['Ticker'])] ### gets prices needed for portfolio
        self.open_positions_df['Last'] = px ### replaces old prices with new prices
        self.open_positions_df['Current Value'] = self.open_positions_df['Last'] * self.open_positions_df['Quantity'] ### updates value of positions
        self.open_positions_df[ "% Gain"] = round((self.open_positions_df['Current Value'] / self.open_positions_df['Basis'])-1, 4)*100 ### calculates new returns
        if cash_add != None: 
            self.cash_add(date = date, amount = cash_add) ### adds cash if needed
        self.snapshots['Positions_{}'.format(date)] = self.open_positions_df.copy() ### update snapshot
        self.snapshots['cash_{}'.format(date)] = self.current_cash.copy() ### update snapshot
        val = self.open_positions_df['Current Value'].sum() + self.current_cash ### calculate portfolio value
        val_ex = self.open_positions_df['Current Value'].sum() ### calculate portfolio value EX cash
        self.snapshots['val_{}'.format(date)] = val # update Snapshot
        tr = pd.DataFrame({'Date':date, 'Value':val, 'Val_ex_cash': val_ex}, index = [0]) # update trackrecord
        self.track_record = pd.concat([self.track_record, tr]).reset_index(drop = True) # update trackrecord
    

