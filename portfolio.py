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
        self.tickers = {}
        self.portfolio_value = value
        self.cash = value

    def get_price(self, date, ticker):
        df = pd.read_csv('assets/historical-symbols/{}.csv'.format(ticker))
        df['Date'] = pd.to_datetime(df['Date'])
        conv_date = datetime.strptime(date, '%y-%m-%d')
        price = df[df['Date'] == conv_date]['Open']
        return price

    def buy(self, purchase_order, date):
        """
        :param purchase_order:{ticker:amount}
        :param date: mm/dd/yyyy
        """
        for p_ticker, p_order in purchase_order:
            p_price = self.get_price(p_ticker)
            self.tickers[p_ticker] += p_order
            self.cash -= p_price * p_order

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        for s_ticker, s_order in sell_order:
            s_price = self.get_price(s_ticker, date)
            self.tickers[s_ticker] -= s_order
            self.cash -= s_price * s_order
