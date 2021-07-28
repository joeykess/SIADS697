import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def create_tracking_dfs():
    results = {}
    for symbol in ['NVDA', 'AMD', 'JPM', 'JNJ', 'MRNA', 'F', 'TSLA', 'MSFT', 'BAC', 'BABA', 'SPY', 'QQQ']:
        df = pd.read_csv(f'assets/short_term_symbols/{symbol}.csv').drop(columns=['Unnamed: 0'])
        df = df.reindex(index=df.index[::-1])
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index(df['time']).drop(columns=['time'])
        df = df.resample('5min').first()
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
        df['volume'] = df['volume'].fillna(0)
        results[symbol] = df
    return results


class portfolio:
    def __init__(self, start_date='2019-07-31', value=1000000, end_date='2021-07-19'):
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
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])
        self.hist_cash_df = pd.DataFrame([[start_date, value]], columns=['Date', 'Cash Available to Trade'])
        self.hist_cash_dict = {datetime.strptime(start_date, '%Y-%m-%d'): value}
        self.tracking_dfs = create_tracking_dfs()

    def reset_account(self):
        self.__init__(self.start_date, self.init_cash, self.end_date)

    def get_price(self, date, ticker):
        """
        :param date: datetime object with time and date '2021-07-19 04:00:00' from a dataframe index
        """
        ticker = ticker.upper()
        return self.tracking_dfs[ticker]

    def buy(self, purchase_order, date):
        """
        put in buy order and returns cannot purchase message if insufficient cash to make purchase
        :param purchase_order:{ticker:shares} - {'MSFT': 10, 'AAPL': 15}
        :param date: mm/dd/yyyy
        """
        for p_ticker, p_order in purchase_order.items():
            p_price = self.get_price(date=date, ticker=p_ticker)
            p_val = p_order * p_price
            if self.current_cash > p_val:
                if p_ticker in self.open_positions_dict:
                    self.open_positions_dict[p_ticker] += p_order
                else:
                    self.open_positions_dict[p_ticker] = p_order
                self.current_cash -= p_val
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'buy',
                     'Ticker': p_ticker, 'Quantity': p_order,
                     'Ticker Value': p_price, 'Total Trade Value': p_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)
        self.hist_cash_dict[datetime.strptime(date, '%Y-%m-%d')] = self.current_cash
        return

    def sell(self, sell_order, date):
        """
        put in sell order as long as we own the position - can't go short yet (will add functionality later)
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
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


def intraday_trading(pf):
    model = keras.models.load_model('assets/models/joey_cnn_intraday/cnn_model.h5')
    model.set_weights('assets/models/joey_cnn_intraday/cnn_weights.h5')


if __name__ == '__main__':
    # commence example portfolio with $1M on 2020-07-19
    ptflio = portfolio(start_date='2020-07-19', value=100000)

