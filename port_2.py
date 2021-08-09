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
def create_sell_dict(ticker_preds,open_positions_df,sell_threshold = -0.05):

    # Establish weights for selling stocks - Not working, only selling full positions
    # How do we deal with cascading price reductions (e.g. only lose 5%, but time after time)
    sell_stocks = ticker_preds[ticker_preds['earn_ratio'] <= sell_threshold]

    sell_dict = {}

    for ticker in sell_stocks.ticker.unique():
        try: 
            # Sell all open positions
            sell_dict[ticker] = open_positions_df[open_positions_df['Ticker']==ticker]['Quantity'].sum()

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
        if t_type.upper() == "SELL" and len(self.open_positions_df)==0: #### if selling before any positions are in the port just exit the func
            return 
        else:
            ###steo 0: update portfolio data:
            if len(self.open_positions_df)==0:
                pass
            else:
                nyse = mcal.get_calendar('NYSE')
                last_update = self.track_record['Date'].iloc[-1]
                today = date
                update_range = list(nyse.valid_days(start_date= last_update, end_date = date))
                update_range = update_range[1:-1]
                for d in update_range:
                    dt = d.strftime('%Y-%m-%d')
                    px = [self.get_price(dt, t) for t in list(self.open_positions_df['Ticker'])] ### gets prices needed for portfolio
                    self.open_positions_df['Last'] = px ### replaces old prices with new prices
                    self.open_positions_df['Current Value'] = self.open_positions_df['Last'] * self.open_positions_df['Quantity'] ### updates value of positions
                    self.open_positions_df[ "% Gain"] = round((self.open_positions_df['Current Value'] / self.open_positions_df['Basis'])-1, 4)*100 ### calculates new returns
                    self.snapshots['Positions_{}'.format(dt)] = self.open_positions_df.copy() ### update snapshot
                    self.snapshots['cash_{}'.format(dt)] = self.current_cash.copy() ### update snapshot
                    val = self.open_positions_df['Current Value'].sum() + self.current_cash ### calculate portfolio value
                    val_ex = self.open_positions_df['Current Value'].sum() ### calculate portfolio value EX cash
                    self.snapshots['val_{}'.format(dt)] = val # update Snapshot
                    tr = pd.DataFrame({'Date':dt, 'Value':val, 'Val_ex_cash': val_ex}, index = [0]) # update trackrecord
                    self.track_record = pd.concat([self.track_record, tr]).reset_index(drop = True) # update trackrecord
                    self.track_record = self.track_record.reset_index(drop = True)
            ###step 1: create an order ticket from the inputs
            trade_ticket= pd.DataFrame(order, index =[0]).T.reset_index().rename(columns = {'index': 'Ticker', 0: 'Quantity'}) ### inserts tickers and quantities into a df
            transaction = [t_type.upper() for i in range(len(trade_ticket))] ### adds transaction type to the trade_ticket df
            px = [self.get_price(date, i) for i in list(trade_ticket['Ticker'])] ### get order prices
            trade_ticket['Last'] = px ### add order prices to trade ticket
            trade_ticket['Trade Type'] = transaction ### add transaction types to trade ticket
            ###step 2: validate orders in trade tickets and add to blotter
            blotter = pd.DataFrame()
            cash_ref = self.current_cash### create a cash counter 
            for t in list(trade_ticket['Ticker']): ###loop through orders
                if t_type.upper() == 'SELL': ### if its an order to sell
                    positions = self.open_positions_df ### pull the portfolio's current positions
                    open_lot = positions[positions['Ticker']==t] ### Isolate the stock in question in the portfolio
                    if len(open_lot)==0: ### If the stock is not in the portfolio - reject
                        print("Rejected Trade: You do not have {} in your portfolio".format(t))
                    elif len(open_lot)>0 and open_lot['Quantity'].iloc[0]< trade_ticket[trade_ticket['Ticker']==t]['Quantity'].iloc[0]: ### check if we are selling more than we own
                        print("Rejected Trade: You do not enough shares of {} in your portfolio".format(t)) ### if so - reject trade
                    else:
                        blotter = pd.concat([blotter, trade_ticket[trade_ticket['Ticker']==t]]) ### if trade checks out print validation
                        trade = blotter[blotter['Ticker'] == t]
                        print("Order to SELL {} shares of {} validated and approved".format(trade['Quantity'].iloc[0], t)) ### add to blotter
                else: #### for BUY Orders
                    trade = trade_ticket[trade_ticket['Ticker']==t] ### Isolate a trade
                    if trade['Quantity'].iloc[0] * trade['Last'].iloc[0] > cash_ref: ### check trade against cash
                        print("Rejected Trade: you do not have enough cash to buy {} shares of {}".format(trade['Quantity'].iloc[0], t)) # If not enough cash reject the trade
                    else:
                        print("Order to BUY {} shares of {} validated and approved".format(trade['Quantity'].iloc[0], t)) #otherwise validate 
                        blotter = pd.concat([blotter, trade]) #add to blotter
                        cash_ref -= (trade['Quantity'].iloc[0] * trade['Last'].iloc[0]) ### adjust for cash 
            blotter['Date'] = [date for i in range(len(blotter))]
            ###step 3: stage and execute trades
            if t_type.upper() == 'SELL':
                blotter['Quantity'] = blotter['Quantity']*-1 ### convert sells negative quantities 
            else:
                pass
            staged_trades = pd.DataFrame() ### executed trades df
            for t in list(blotter['Ticker']): ### loop through orders
                ex_trade = blotter[blotter['Ticker']==t] ### grab an order
                ex_trade['Basis'] = ex_trade['Quantity'] * ex_trade['Last'] ### establish cost basis
                ex_trade['Current Value'] = ex_trade['Quantity'] * ex_trade['Last'] ### establish current value
                ex_trade['Purchase Price'] = ex_trade['Last'] ### lock in purchase price for this lot
                staged_trades = pd.concat([staged_trades, ex_trade]) ### add to staged trades 
            self.hist_trades_df = pd.concat([self.hist_trades_df, staged_trades]) ### record trades in historic activity
            staged_trades = staged_trades.drop('Trade Type', axis = 1) ### Prep for execution
            self.open_positions_df = pd.concat([self.open_positions_df, staged_trades]) ### Buy/Sell
            self.current_cash -= staged_trades['Basis'].sum() ### Adjust cash
            self.open_positions_df['% Gain'] = round((self.open_positions_df['Current Value']/self.open_positions_df['Basis'])-1,4)*100 ### current gain 
            ###step 3: consolidate open_positions:
            exp_portfolio = self.open_positions_df
            consolidated_port = pd.DataFrame()
            for t in exp_portfolio['Ticker'].unique():
                open_lots = exp_portfolio[exp_portfolio['Ticker']==t]
                dt = open_lots['Date'].min()
                quant = open_lots['Quantity'].sum()
                basis = open_lots['Basis'].sum()
                ppx = basis/quant
                last = self.get_price(date = date, ticker = t)
                cv = quant * last
                gain = round((cv/basis)-1,4)*100
                consolidated_line = pd.DataFrame({'Date':dt,'Ticker':t, 'Quantity':quant,'Basis':basis,'Purchase Price':ppx, "Current Value":cv, "Last":last, "% Gain":gain}, index = [0])
                consolidated_port = pd.concat([consolidated_port, consolidated_line])
            consolidated_port = consolidated_port[consolidated_port['Quantity']!=0]
            self.open_positions_df = consolidated_port.reset_index(drop = True)
            val = self.open_positions_df["Current Value"].sum()+self.current_cash
            val_ex = self.open_positions_df["Current Value"].sum()
            tr = pd.DataFrame({'Date':date, 'Value':val, 'Val_ex_cash': val_ex}, index = [0])
            self.track_record = pd.concat([self.track_record, tr]).reset_index(drop=True)
            self.snapshots['Positions_{}'.format(date)] = self.open_positions_df ### update snapshot
            self.snapshots['cash_{}'.format(date)] = self.current_cash ### update snapshot
            self.snapshots['val_{}'.format(date)] = val # update Snapshot
            print('Trades Executed')

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
        nyse = mcal.get_calendar('NYSE')
        last_update = self.track_record['Date'].iloc[-1]
        today = date
        update_range = list(nyse.valid_days(start_date= last_update, end_date = date))
        update_range = update_range[1:-1]
        for d in update_range:
            dt = d.strftime('%Y-%m-%d')
            px = [self.get_price(dt, t) for t in list(self.open_positions_df['Ticker'])] ### gets prices needed for portfolio
            self.open_positions_df['Last'] = px ### replaces old prices with new prices
            self.open_positions_df['Current Value'] = self.open_positions_df['Last'] * self.open_positions_df['Quantity'] ### updates value of positions
            self.open_positions_df[ "% Gain"] = round((self.open_positions_df['Current Value'] / self.open_positions_df['Basis'])-1, 4)*100 ### calculates new returns
            self.snapshots['Positions_{}'.format(dt)] = self.open_positions_df.copy() ### update snapshot
            self.snapshots['cash_{}'.format(dt)] = self.current_cash.copy() ### update snapshot
            val = self.open_positions_df['Current Value'].sum() + self.current_cash ### calculate portfolio value
            val_ex = self.open_positions_df['Current Value'].sum() ### calculate portfolio value EX cash
            self.snapshots['val_{}'.format(dt)] = val # update Snapshot
            tr = pd.DataFrame({'Date':dt, 'Value':val, 'Val_ex_cash': val_ex}, index = [0]) # update trackrecord
            self.track_record = pd.concat([self.track_record, tr]).reset_index(drop = True) # update trackrecord
            self.track_record = self.track_record.reset_index(drop = True)
        return 
    

