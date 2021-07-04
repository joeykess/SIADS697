import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt


def candle_charts(ticker, start, end):
    df2 = pd.read_csv(f'assets/historical-symbols/{ticker}.csv')
    df2['MA15'] = df2['Close'].rolling(window=15).mean()

    def bollinger_bands(data, sma, window):
        std = data.rolling(window=window).std()
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        return upper_bb, lower_bb

    df2['upper_bb'], df2['lower_bb'] = bollinger_bands(df2['Close'], df2['MA15'], 15)
    df2['ema15'] = df2['Close'].ewm(span=15).mean()
    df = df2.loc[(df2['Date'] >= start) & (df2['Date'] <= end)].copy()
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    df = df.set_index(['Date'])
    fig, ax = plt.subplots(figsize=(12, 6))
    mpf.plot(df, type='candlestick', ax=ax)
    df.reset_index().plot(kind='line', x='Date', y='upper_bb', ax=ax, legend=None)
    ax.set_facecolor('white')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_ylabel("")
    plt.show()


if __name__ == '__main__':
    candle_charts('BA', '2015-01-01', '2015-01-22')
