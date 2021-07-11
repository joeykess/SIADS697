import pandas as pd
from datetime import datetime
import os
import re


def create_running_df():
    path_to_files = 'assets/historical-symbols'
    df = pd.DataFrame()
    for file in os.listdir(path_to_files):
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
        # self.tracking_df = create_running_df()
        self.hist_trades_dict = {}
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])

    def _reset_account(self):
        self.cash = self.init_cash
        self.equity = self.current_cash

    def get_price(self, date, ticker):
        df = pd.read_csv('assets/historical-symbols/{}.csv'.format(ticker))
        df['Date'] = pd.to_datetime(df['Date'])
        conv_date = datetime.strptime(date, '%Y-%m-%d')
        # assume trades are set in the middle of OHLC
        price = df[df['Date'] == conv_date][['Open', 'High', 'Low', 'Close']].mean(axis=1).values[0]
        return price

    def buy(self, purchase_order, date):
        """
        :param purchase_order:{ticker:amount}
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
                if date in self.hist_trades_dict:
                    self.hist_trades_dict[date].append(['buy', p_ticker, p_order, p_price,
                                                   p_val, self.current_cash])
                else:
                    self.hist_trades_dict[date] = [['buy', p_ticker, p_order, p_price,
                                               p_val, self.current_cash]]
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'buy',
                     'Ticker': p_ticker, 'Quantity': p_order,
                     'Ticker Value': p_price, 'Total Trade Value': p_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)
            else:
                return "Cannot purchase - low funds"
        return

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        for s_ticker, s_order in sell_order.items():
            s_price = self.get_price(date=date, ticker=s_ticker)
            s_val = s_order * s_price
            if s_ticker in self.open_positions_dict:
                self.open_positions_dict[s_ticker] += s_order
                self.current_cash += s_val
                if date in self.hist_trades_dict:
                    self.hist_trades_dict[date].append(['sell', s_ticker, s_order, s_price,
                                                   s_val, self.current_cash])
                else:
                    self.hist_trades_dict[date] = [['sell', s_ticker, s_order, s_price,
                                               s_val, self.current_cash]]
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'sell',
                     'Ticker': s_ticker, 'Quantity': s_order,
                     'Ticker Value': s_price, 'Total Trade Value': s_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)

    def view_trade_history(self):
        return self.hist_trades_df


if __name__ == '__main__':
    ptflio = portfolio(start_date='2015-01-01', value=1000000)
    ptflio.buy(purchase_order={'MSFT': 100, 'AAPL': 100}, date='2015-10-05')
    ptflio.sell(sell_order={'MSFT': 10, 'AAPL': 10}, date='2015-10-06')
    print(ptflio.view_trade_history())
