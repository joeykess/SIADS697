# add class for the whole portfolio
# basically can track whole portfolio performance
# convert everything into shares (round numbers)
# portfolio will have list of stocks (passed in)
# based off price and weights we're using, trading bot will do whatever it wants to do
# dump and re-balance every x_months
import pandas as pd


class portfolio:
    def __init__(self, value):
        self.tickers = {}
        self.portfolio_value = value
        self.cash = value

    def buy(self, purchase_order, date):
        """
        :param purchase_order:{ticker:amount}
        :param date: mm/dd/yyyy
        """
        for p_ticker, p_price in purchase_order:
            df = pd.read_csv('assets/historical-symbols/{}.csv'.format(p_ticker))

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        for s_ticker, s_price in sell_order:
            df = pd.read_csv('assets/historical-symbols/{}.csv'.format(s_ticker))
