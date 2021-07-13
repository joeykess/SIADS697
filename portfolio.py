import pandas as pd
from datetime import datetime
import os
import re


def create_running_df():
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
        self.end_date = end_date
        self.start_date = start_date
        self.init_cash = value
        self.current_portfolio_value = value
        self.current_cash = value
        self.open_positions_dict = {}
        self.open_positions_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        self.tracking_df = create_running_df()
        self.tracking_df.index = pd.to_datetime(self.tracking_df.index)
        self.hist_trades_dict = {}
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])
        self.hist_cash_df = pd.DataFrame([[start_date, value]], columns=['Date', 'Cash Available to Trade'])
        self.hist_cash_dict = {datetime.strptime(start_date, '%Y-%m-%d'): value}

    def reset_account(self):
        self.hist_trades_dict = {}
        self.current_cash = pd.DataFrame([[self.start_date, self.init_cash]],
                                         columns=['Date', 'Cash Available to Trade'])
        self.open_positions_df = pd.DataFrame(columns=['Date', 'Ticker', 'Quantity'])
        self.hist_trades_df = pd.DataFrame(columns=[
            'Date', 'Order Type', 'Ticker', 'Quantity', 'Ticker Value', 'Total Trade Value', 'Remaining Cash'
        ])

    def get_price(self, date, ticker):
        ticker = ticker.upper()
        conv_date = datetime.strptime(date, '%Y-%m-%d')
        return self.tracking_df.loc[conv_date][ticker]

    def buy(self, purchase_order, date):
        """
        :param purchase_order:{ticker:shares} - {'MSFT': 10, 'AAPL': 15}
        :param date: mm/dd/yyyy
        """
        if date not in self.hist_trades_dict:
            if len(self.hist_trades_dict) == 0:
                self.hist_trades_dict[date] = {i: 0 for i in self.tracking_df.columns.tolist()}
            else:
                self.hist_trades_dict[date] = self.hist_trades_dict[list(self.hist_trades_dict.keys())[-1]].copy()
        for p_ticker, p_order in purchase_order.items():
            p_price = self.get_price(date=date, ticker=p_ticker)
            p_val = p_order * p_price
            if self.current_cash > p_val:
                if p_ticker in self.open_positions_dict:
                    self.open_positions_dict[p_ticker] += p_order
                else:
                    self.open_positions_dict[p_ticker] = p_order
                self.hist_trades_dict[date][p_ticker] += p_order
                self.current_cash -= p_val
                self.hist_trades_df = self.hist_trades_df.append(
                    {'Date': date, 'Order Type': 'buy',
                     'Ticker': p_ticker, 'Quantity': p_order,
                     'Ticker Value': p_price, 'Total Trade Value': p_val,
                     'Remaining Cash': self.current_cash}, ignore_index=True)
            else:
                return "Cannot purchase - low funds"
        self.hist_cash_dict[datetime.strptime(date, '%Y-%m-%d')] = self.current_cash
        return

    def sell(self, sell_order, date):
        """
        :param sell_order: {ticker:amount}
        :param date: mm/dd/yyyy
        """
        if date not in self.hist_trades_dict:
            self.hist_trades_dict[date] = self.hist_trades_dict[list(self.hist_trades_dict.keys())[-1]].copy()
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

    def view_trade_history(self):
        return self.hist_trades_df.set_index('Date')

    def calculate_daily_returns(self, end_date='2021-07-02'):
        """
        :param end_date: date is hard coded to last date in data
        """
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        hist_df = self.tracking_df[self.tracking_df.index <= end_date]
        hist_df = hist_df[hist_df.index >= start_date]
        hist_df = self.hist_trades_df.set_index('Date')
        hist_df.index = pd.to_datetime(hist_df.index)
        cash_df = pd.DataFrame.from_dict(self.hist_cash_dict, orient='index', columns=['Cash Available to Trade'])
        cash_df = cash_df.reindex(pd.date_range(self.start_date, end_date, freq='B')).ffill()
        cash_df.index = pd.to_datetime(cash_df.index)
        ticker_df = pd.DataFrame.from_dict(self.hist_trades_dict, orient='index')
        ticker_df.index = pd.to_datetime(ticker_df.index)
        ticker_df = ticker_df.reindex(pd.date_range(self.start_date, end_date, freq='B')).ffill()
        ticker_df = ticker_df.mul(self.tracking_df, fill_value=0)
        ticker_df = ticker_df.fillna(0).loc[start_date:end_date]
        ticker_vals = pd.DataFrame(ticker_df.sum(axis=1), columns=['Stock Value'])
        returns_df = pd.concat([ticker_vals, cash_df], axis=1)
        returns_df['Total Portfolio Value'] = returns_df['Stock Value'] + returns_df['Cash Available to Trade']
        returns_df['Rate of Return'] = returns_df['Total Portfolio Value'].apply(lambda x: (x - self.init_cash) / x)
        returns_df['Rate of Return'] = returns_df['Rate of Return'].apply(lambda x: "{0:.2f}%".format(x*100))
        return returns_df


if __name__ == '__main__':
    ptflio = portfolio(start_date='2015-01-05', value=100000)
    ptflio.buy(purchase_order={'ZTS': 100, 'AAPL': 100}, date='2015-01-05')
    ptflio.buy(purchase_order={'AAL': 100, 'ZION': 100}, date='2015-10-05')
    ptflio.sell(sell_order={'ZTS': 100, 'AAPL': 100}, date='2015-10-09')
    ptflio.sell(sell_order={'AAL': 100, 'ZION': 100}, date='2015-10-09')
    print(ptflio.calculate_daily_returns())
    ptflio.reset_account()
