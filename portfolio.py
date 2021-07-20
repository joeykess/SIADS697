import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import os
import re
import warnings
import plotly.express as px

warnings.simplefilter(action='ignore', category=FutureWarning)


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
                    temp_df[symbol] = temp_df[['Close', 'Open', 'High', 'Low']].mean(axis=1)
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
        self.open_positions_dict = {}
        self.open_positions_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity', 'Price'])
        self.tracking_df = create_running_df()
        self.tracking_df.index = pd.to_datetime(self.tracking_df.index)
        self.hist_trades_dict = {}
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])
        self.hist_cash_df = pd.DataFrame([[start_date, value]], columns=['Date', 'Cash Available to Trade'])
        self.hist_cash_dict = {datetime.strptime(start_date, '%Y-%m-%d'): value}

    def reset_account(self):
        self.__init__(self.start_date, self.init_cash, self.end_date)

    def get_price(self, date, ticker):
        ticker = ticker.upper()
        conv_date = datetime.strptime(date, '%Y-%m-%d')
        return self.tracking_df.loc[conv_date][ticker]

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

    def buy(self, purchase_order, date):
        """
        put in buy order and returns cannot purchase message if insufficient cash to make purchase
        :param purchase_order:{ticker:shares} - {'MSFT': 10, 'AAPL': 15}
        :param date: mm/dd/yyyy
        """
        # brings trading dictionary up to date with what was previously held
        if date not in self.hist_trades_dict:
            if len(self.hist_trades_dict) == 0:
                self.hist_trades_dict[date] = {i: 0 for i in self.tracking_df.columns.tolist()}
            else:
                self.hist_trades_dict[date] = self.hist_trades_dict[list(self.hist_trades_dict.keys())[-1]].copy()
        # goes through order and buys all necessary stocks
        # returns issue string if not enough cash present

        ### DO WE WANT TO EXPAND THIS STRING SUCH THAT WE DON'T ORDER ANYTHING IF WE CAN'T ORDER EVERYTHING??
        for p_ticker, p_order in purchase_order.items():
            p_price = self.get_price(date=date, ticker=p_ticker)
            p_val = p_order * p_price
            if self.current_cash > p_val:
                if p_ticker in self.open_positions_dict:
                    self.open_positions_dict[p_ticker] += p_order

                    # Adding current order to open_positions_df - WILL NOT WORK FOR MORE FREQUENT THAN DAILY
                    self.open_positions_df = self.open_positions_df.append({'Date':date,'Ticker':p_ticker,
                                                   'Quantity':p_order,'Price':p_price},ignore_index=True)

                else:
                    self.open_positions_dict[p_ticker] = p_order
                    # Adding current order to open_positions_df - WILL NOT WORK FOR MORE FREQUENT THAN DAILY
                    self.open_positions_df = self.open_positions_df.append({'Date':date,'Ticker':p_ticker,
                                                   'Quantity':p_order,'Price':p_price},ignore_index=True)
                self.hist_trades_dict[date][p_ticker] += p_order
                self.current_cash -= p_val
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'buy',
                     'Ticker': p_ticker, 'Quantity': p_order,
                     'Ticker Value': p_price, 'Total Trade Value': p_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)
            else:
                return f"Cannot purchase {p_order} shares of {p_ticker} for a total of ${p_val} because current cash is ${self.current_cash}"
        self.hist_cash_dict[datetime.strptime(date, '%Y-%m-%d')] = self.current_cash
        return

    def sell(self, sell_order, date):
        """
        put in sell order as long as we own the position - can't go short yet (will add functionality later)
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        if date not in self.hist_trades_dict:
            if len(self.hist_trades_dict) == 0:
                self.hist_trades_dict[date] = {i: 0 for i in self.tracking_df.columns.tolist()}
            else:
                self.hist_trades_dict[date] = self.hist_trades_dict[list(self.hist_trades_dict.keys())[-1]].copy()

        # goes through order and sells all necessary stocks
        for s_ticker, s_order in sell_order.items():
            s_price = self.get_price(date=date, ticker=s_ticker)
            s_val = s_order * s_price
            if s_ticker in self.open_positions_dict:
                self.hist_trades_dict[date][s_ticker] -= s_order
                self.open_positions_dict[s_ticker] -= s_order
                self.current_cash += s_val
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'sell',
                     'Ticker': s_ticker, 'Quantity': s_order,
                     'Ticker Value': s_price, 'Total Trade Value': s_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)

            else:
                print(f'You do not own {s_ticker}')
        self.hist_cash_dict[datetime.strptime(date, '%Y-%m-%d')] = self.current_cash

    def sell_open_position(self, tickers, date):
        """
        sells all open positions for specific tickers
        :param tickers: list of tickers - or single ticker as string
        :param date: date selling
        """
        if type(tickers) == str:
            self.sell(sell_order=self.open_positions_dict[tickers], date=date)
        else:
            for ticker in tickers:
                self.sell(sell_order=self.open_positions_dict[ticker], date=date)

    def sell_all(self, date):
        """
        liquidates all open positions
        """
        self.sell(sell_order=self.open_positions_dict, date=date)

    def view_trade_history(self):
        return self.hist_trades_df.set_index('Date')

    def calculate_daily_returns(self, end_date='2021-07-02'):
        """
        :param end_date: date is hard coded to last date in data
        """
        # get NYSE calendar to find days which aren't traded and then takes that calendar and adds it to other dataset
        nyse = mcal.get_calendar('NYSE')
        early = nyse.schedule(start_date=self.start_date, end_date=end_date)
        d_range = mcal.date_range(early, frequency='1D').normalize().date
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        # gets all historical data from starting date to ending date and then sets index to date in datetime format
        hist_df = self.tracking_df[self.tracking_df.index <= end_date]
        hist_df = hist_df[hist_df.index >= start_date]
        hist_df = self.hist_trades_df.set_index('Date')
        hist_df.index = pd.to_datetime(hist_df.index)
        # gets all cash extended out from all trading days
        cash_df = pd.DataFrame.from_dict(self.hist_cash_dict, orient='index', columns=['Cash Available to Trade'])
        cash_df = cash_df.reindex(pd.date_range(self.start_date, end_date, freq='B')).ffill()
        cash_df.index = pd.to_datetime(cash_df.index)
        # gets all historical trades
        ticker_df = pd.DataFrame.from_dict(self.hist_trades_dict, orient='index')
        ticker_df.index = pd.to_datetime(ticker_df.index)
        ticker_df = ticker_df.reindex(pd.date_range(start_date, end_date, freq='B')).ffill()
        # muted FutureWarning: Indexing a timezone-naive DatetimeIndex with a timezone-aware datetime is
        # deprecated and will raise KeyError in a future version.  Use a timezone-naive object instead.
        ticker_df.index = pd.DatetimeIndex(ticker_df.index)
        ticker_df.index = ticker_df.index.normalize()
        ticker_df = ticker_df.loc[d_range]
        # multiplies the stock value at whatever date to how many you own
        ticker_df = ticker_df.mul(self.tracking_df, fill_value=0)
        ticker_df = ticker_df.fillna(0).loc[start_date:end_date]
        ticker_vals = pd.DataFrame(ticker_df.sum(axis=1), columns=['Stock Value'])
        # concats the stock value and cash dataframes together then does easy math to get ROI and total value
        returns_df = pd.concat([ticker_vals, cash_df], axis=1)
        returns_df['Total Portfolio Value'] = returns_df['Stock Value'] + returns_df['Cash Available to Trade']
        returns_df['Rate of Return'] = returns_df['Total Portfolio Value'].apply(lambda x: (x - self.init_cash) / x)
        returns_df['Rate of Return'] = returns_df['Rate of Return'].apply(lambda x: "{0:.2f}%".format(x * 100))
        returns_df = returns_df.dropna()

        # formatting from Jeff's figures
        returns_fig = px.line(returns_df['Rate of Return'].apply(lambda x: float(str(x).strip('%')) / 100),
                              title='Returns on Initial Investment', width=700, height=500)
        returns_fig.update_layout(width=950, height=300,
                                  margin=dict(l=20, r=20, t=50, b=10),
                                  paper_bgcolor='white',
                                  plot_bgcolor='white',
                                  legend=dict(orientation="h"),
                                  yaxis_tickformat='.2%',
                                  title=dict(text='Performance Chart', font=dict(size=20, color='black'), x=0.5,
                                             y=0.96))
        returns_fig.show()
        return returns_df


if __name__ == '__main__':
    # commence example portfolio with $1M on 2015-01-05
    # will purchase AAPL at various amounts and then use sell all function to liquidate account of all shares
    # buy/sell with order format x.buy({ticker:amount}, date=trading_date_only_please)
    # best way to use this with a model is to loop through trading dates within start and end date that you choose
    # example-Joey will test his model with unseen data (only training on pre-2020, and testing on 2020-present)
    ptflio = portfolio(start_date='2015-01-05', value=100000)
    ptflio.buy(purchase_order={'AAPL': 100}, date='2015-01-05')
    ptflio.buy(purchase_order={'AAPL': 2}, date='2020-5-26')
    ptflio.buy(purchase_order={'AAPL': 2}, date='2020-8-24')
    ptflio.buy(purchase_order={'AAPL': 2}, date='2020-10-02')
    ptflio.buy(purchase_order={'AAPL': 2}, date='2020-11-02')
    ptflio.buy(purchase_order={'AAPL': 2}, date='2020-12-22')
    ptflio.sell(sell_order={'AAPL': 2}, date='2020-12-22')
    ptflio.sell_all(date='2020-12-22')
    print(ptflio.calculate_daily_returns())
    ptflio.reset_account()
