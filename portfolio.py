# add class for the whole portfolio
# basically can track whole portfolio performance
# convert everything into shares (round numbers)
# portfolio will have list of stocks (passed in)
# based off price and weights we're using, trading bot will do whatever it wants to do

# dump and re-balance every x_months
import pandas as pd
from datetime import datetime


class portfolio:
    def __init__(self, value):
        self.current_positions = {}
        self.init_cash = value
        self.portfolio_value = value
        self.current_cash = value
        self.hist_trades = {}
        # hist_trades is in the form of [buy/sell, ticker, quantity, price, and remaining cash]

    def _reset_account(self):
        self.cash = self.init_cash
        self.equity = self.current_cash

    def get_price(self, date, ticker):
        df = pd.read_csv('assets/historical-symbols/{}.csv'.format(ticker))
        df['Date'] = pd.to_datetime(df['Date'])
        conv_date = datetime.strptime(date, '%y-%m-%d')
        # assume trades are set in the middle of OHLC
        price = df[df['Date'] == conv_date][['Open', 'High', 'Low', 'Close']].mean(axis=1)
        return price

    def buy(self, purchase_order, date):
        """
        :param purchase_order:{ticker:amount}
        :param date: mm/dd/yyyy
        """
        for p_ticker, p_order in purchase_order:
            p_price = self.get_price(date=date, ticker=p_ticker)
            if self.current_cash > p_price * p_order:
                self.current_positions[p_ticker] += p_order
                self.current_cash -= p_price * p_order
                self.hist_trades[date] = ['buy', p_ticker, p_order, p_price, p_order*p_price, self.current_cash]
            else:
                return 'Cannot purchase - low funds'
        return

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        for s_ticker, s_order in sell_order:
            s_price = self.get_price(s_ticker, date)
            self.current_positions[s_ticker] -= s_order
            self.current_cash += s_price * s_order

    def view_portfolio_history(self):
        return pd.DataFrame.from_dict(self.hist_trades, orient='index', columns=[
            'Order Type', 'Ticker', 'Quantity', 'Value', 'Total Trade Value', 'Remaining Cash'
        ])
