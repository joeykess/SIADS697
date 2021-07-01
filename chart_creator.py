import plotly.graph_objects as go
import pandas as pd


def candle_charts(ticker, start, end):
    df = pd.read_csv(f'assets/historical-symbols/{ticker}.csv')
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.show()


if __name__ == '__main__':
    candle_charts('BA', '2015-01-01', '2015-01-31')
