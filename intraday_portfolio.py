import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal
import plotly.express as px
import mplfinance as mpf
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
from keras.preprocessing import image
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def create_tracking_df():
    if os.path.exists('assets/short_term_symbols/total.csv'):
        results = pd.read_csv('assets/short_term_symbols/total.csv')
        results['time'] = pd.to_datetime(results['time'])
        results = results.set_index(results['time']).drop(columns=['time'])
    else:
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
        results = pd.concat([v.add_prefix(f'{k}_') for k, v in results.items()], axis=1, join='outer')
        for symb in ['NVDA', 'AMD', 'JPM', 'JNJ', 'MRNA', 'F', 'TSLA', 'MSFT', 'BAC', 'BABA', 'SPY', 'QQQ']:
            results[[f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']] = results[
                [f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']].ffill()
            results[f'{symb}_volume'] = results[f'{symb}_volume'].fillna(0)
            results[[f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']] = results[
                [f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']].bfill()
        results = results.between_time('4:05', '20:00')
        results.to_csv('assets/short_term_symbols/total.csv')
    return results


def bollinger_bands(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb


def create_candles(plot_df, file):
    if not os.path.exists('assets/models/joey_cnn_intraday/live_test'):
        os.mkdir('assets/models/joey_cnn_intraday/live_test')
    fig, ax = plt.subplots(figsize=(5, 5))
    mpf.plot(plot_df, type='candlestick', style='charles', ax=ax)
    plot_df.reset_index().plot(kind='line', y='upper_bb', color='blue', lw=3, alpha=0.75, ax=ax, legend=None)
    plot_df.reset_index().plot(kind='line', y='lower_bb', color='orange', lw=3, alpha=0.75, ax=ax, legend=None)
    plot_df.reset_index().plot(kind='line', y='ema12', color='black', lw=3, alpha=0.75, ax=ax, legend=None)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.axis('off')
    save_spot = f'assets/models/joey_cnn_intraday/live_test'
    if not os.path.exists(save_spot):
        os.makedirs(save_spot)
    plt.savefig(f'{save_spot}/{file}.png', dpi=50, bbox_inches='tight')
    plt.close()


class intraday_portfolio:
    def __init__(self, start_date='2020-07-20', value=1000000, end_date='2021-07-20'):
        """
        :param start_date: beginning date to start portfolio
        :param value: amount of initial cash to use to buy shares
        :param end_date: end date of trading simulation - 2021-07-20 is last day in data so that is hard coded for now
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
        self.tracking_df = create_tracking_df()

    def reset_account(self):
        self.__init__(self.start_date, self.init_cash, self.end_date)

    def get_price(self, date, symb):
        """
        :param symb: ticker string
        :param date: datetime object with time and date '2021-07-19 04:00:00' from a dataframe index
        """
        symb = symb.upper()
        try:
            return self.tracking_df.loc[date][[f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']].mean()
        except:
            return -1

    def buy(self, purchase_order, date):
        """
        put in buy order and returns cannot purchase message if insufficient cash to make purchase
        :param purchase_order:{ticker:shares} - {'MSFT': 10, 'AAPL': 15}
        :param date: mm/dd/yyyy
        """
        for p_ticker, p_order in purchase_order.items():
            p_price = self.get_price(date=date, symb=p_ticker)
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
            s_price = self.get_price(date=date, symb=s_ticker)
            s_val = s_order * s_price
            if s_ticker in self.open_positions_dict:
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


def intraday_trading(pf, model_path, weights_path):
    # model = keras.models.load_model(model_path)
    # model.set_weights(weights_path)

    start_date = '2020-07-20 04:05:00'
    # loop through array above in portfolio (pf)
    df = pf.tracking_df.loc[start_date:]
    for time in tqdm(range(0, df.shape[0])):
        # for each trading period, after trading delete folder contents
        # get predictions here
        trading_time = df.iloc[time].name.hour >= 5 and df.iloc[time].name.minute % 15 == 0
        closing = df.iloc[time].name.minute >= 0 and df.iloc[time].name.hour >= 20
        if trading_time and not closing:
            pf.sell_all(time)
            if os.path.exists('assets/models/joey_cnn_intraday/live_test'):
                shutil.rmtree('assets/models/joey_cnn_intraday/live_test')
            tickers = ['NVDA', 'AMD', 'JPM', 'JNJ', 'MRNA', 'F', 'TSLA', 'MSFT', 'BAC', 'BABA', 'SPY', 'QQQ']
            timed_df = df.iloc[time - 12: time].copy()
            for symb in tickers:
                symbol_df = timed_df[[f'{symb}_open', f'{symb}_high', f'{symb}_low', f'{symb}_close']]
                symbol_df = symbol_df.rename(columns={f'{symb}_open': 'open', f'{symb}_high': 'high',
                                                      f'{symb}_low': 'low', f'{symb}_close': 'close'})
                if 0.0 in symbol_df['volume'].value_counts():
                    if symbol_df['volume'].value_counts()[0.0] <= 4:
                        create_candles(plot_df=symbol_df, file=str(tickers.index(symb)))
                else:
                    create_candles(plot_df=symbol_df, file=str(tickers.index(symb)))
            images = []
            path = 'assets/models/joey_cnn_intraday/live_test'
            width, height = 203, 202
            # gets all images and stacks them together for predictions
            for img in os.listdir(path):
                img = os.path.join(path, img)
                img = image.load_img(img, target_size=(width, height))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images.append(img)
            images = np.vstack(images)
            preds = model.predict_classes(images, batch_size=12)  # will output 1 for buy and 0 for hold
            cash = pf.current_cash
            preds_buy = [tickers[i] for i, x in enumerate(preds) if x == 1]
            buy_order = {}
            for item in preds_buy:
                max_cost = cash * 0.75  # trading max 75% cash every trade
                cost_per = max_cost / len(preds_buy)
                price = pf.get_price(date=time, symb=item)
                buy_order['item'] = int(np.floor(cost_per / price))
            pf.buy(purchase_order=buy_order, date=time)
        if closing:
            pf.sell_all(time)


if __name__ == '__main__':
    # commence example portfolio with $100k on 2020-07-20
    model = 'assets/models/joey_cnn_intraday/cnn_model_100epochs_2classes.h5'
    weights = 'assets/models/joey_cnn_intraday/cnn_weights_100epochs_2classes.h5'
    ptflio = intraday_portfolio(start_date='2020-07-20', value=100000)
    intraday_trading(ptflio, model_path=model, weights_path=weights)
