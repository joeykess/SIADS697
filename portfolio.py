import pandas as pd
from datetime import datetime
import pandas as pd
from datetime import datetime
import os
import re
from tqdm import tqdm


def create_running_df():
    path_to_files = 'assets/historical-symbols'
    df = pd.DataFrame()
    for file in tqdm(os.listdir(path_to_files)):
        try:
            if file.endswith('.csv'):
                symbol = re.findall(r'(.*).csv', file)[0]
                temp_df = pd.read_csv(path_to_files + '/' + file).set_index(['Date'])
                temp_df[symbol] = temp_df['Close']
                if df.shape[0] == 0:
                    df = temp_df[symbol].to_frame(name=symbol)
                else:
                    df = df.join(temp_df[symbol].to_frame(name=symbol))
        except Exception as e:
            print(e)
    return df


class portfolio:
    def __init__(self, start_date, value):
        self.open_positions_dict = {}
        self.start_date = start_date
        self.init_cash = value
        self.portfolio_value = value
        self.current_cash = value
        self.open_positions_df = pd.DataFrame()
        self.tracking_df = create_running_df()

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
                self.open_positions_dict[p_ticker] += p_order
                self.current_cash -= p_price * p_order
                self.hist_trades[date] = ['buy', p_ticker, p_order, p_price, p_order * p_price, self.current_cash]
            else:
                return "Cannot purchase - low funds"
        return

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        for s_ticker, s_order in sell_order:
            s_price = self.get_price(s_ticker, date)
            self.open_positions_dict[s_ticker] -= s_order
            self.current_cash += s_price * s_order

    def view_portfolio_history(self):
        return pd.DataFrame.from_dict(self.hist_trades, orient='index', columns=[
            'Order Type', 'Ticker', 'Quantity', 'Value', 'Total Trade Value', 'Remaining Cash'
        ])
