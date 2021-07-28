import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def bollinger_bands(data, sma, window):
    std = data.rolling(window=window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb


def create_candles(plot_df, folder, file):
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
    save_spot = f'assets/cnn_images/{folder}/'
    if not os.path.exists(save_spot):
        os.makedirs(save_spot)
    plt.savefig(f'{save_spot}/{file}.png', dpi=50, bbox_inches='tight')
    plt.close('all')


def main(ticker, num_days):
    df = pd.read_csv(f'assets/short_term_symbols/{ticker}.csv').drop(columns=['Unnamed: 0'])
    df = df.reindex(index=df.index[::-1])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index(df['time']).drop(columns=['time'])
    df = df.resample('5min').first()
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
    df['volume'] = df['volume'].fillna(0)
    df['MA12'] = df['close'].rolling(window=12).mean()
    df['upper_bb'], df['lower_bb'] = bollinger_bands(df['close'], df['MA12'], 12)
    df['ema12'] = df['close'].ewm(span=12).mean()
    df = df[(df.index < '2020-07-19')]
    df = df.iloc[11:]
    for start in range(0, df.shape[0], num_days):
        data = df.iloc[start:start + num_days].copy()
        # when a quarter the data is filled in skip for training purpose
        last_point = data[['open', 'high', 'low', 'close']].iloc[0].mean()
        next_points = [last_point]
        # checks within next 2 hours if going up or down (if no volume present)
        for index, row in df.iloc[start + 1:start + 24].iterrows():
            if row['volume'] > 0.0 and len(next_points) <= 3:
                next_points.append(row[['open', 'high', 'low', 'close']].mean())
        result = []
        for ind, j in enumerate(next_points[:-1]):
            n = next_points[ind + 1]
            if n > j:
                result.append('up')
            elif j == n:
                result.append('same')
            else:
                result.append('down')
        folder = '_'.join(result)
        file = ticker + '_' + str(start)
        if 0.0 in data['volume'].value_counts():
            if data['volume'].value_counts()[0.0] <= 4:
                create_candles(data, folder, file)
        else:
            create_candles(data, folder, file)


if __name__ == '__main__':
    symbol_list = ['NVDA', 'AMD', 'JPM', 'JNJ', 'MRNA', 'F', 'TSLA', 'MSFT', 'BAC', 'BABA', 'SPY', 'QQQ']
    if not os.path.exists('assets/cnn_images'):
        os.makedirs('assets/cnn_images')
    for symbol in tqdm(symbol_list):
        try:
            main(ticker=symbol, num_days=12)
        except Exception as e:
            print(e)
