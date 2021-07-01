import plotly.graph_objects as go
import pandas as pd


def candle_charts(ticker, start, end):
    df2 = pd.read_csv(f'assets/historical-symbols/{ticker}.csv')
    df2['MA15'] = df2['Close'].rolling(window=15).mean()

    def bollinger_bands(data, sma, window):
        std = data.rolling(window=window).std()
        upper_bb = sma + std * 2
        lower_bb = sma - std * 2
        return upper_bb, lower_bb

    df2['upper_bb'], df2['lower_bb'] = bollinger_bands(df2['Close'], df2['MA15'], 15)
    df = df2[(df2['Date'] >= start) & (df2['Date'] <= end)]
    candle = {
        "name": "GS",
        "type": "candlestick",
        "x": df['Date'],
        "yaxis": "y2",
        "low": df['Low'],
        "high": df['High'],
        "open": df['Open'],
        "close": df['Close'],
        "decreasing": {"line": {"color": "#7F7F7F"}},
        "increasing": {"line": {"color": "#17BECF"}}
    }
    moving_avg = {
        "line": {"width": 3},
        "mode": "lines",
        "name": "Moving Average",
        "type": "scatter",
        "x": df['Date'],
        "y": df['MA15'],
        "yaxis": "y2",
        "marker": {"color": "#E377C2"}
    }
    bband_up = {
        "line": {"width": 3},
        "name": "Bollinger Bands",
        "mode": "lines",
        "type": "scatter",
        "x": df['Date'],
        "y": df['upper_bb'],
        "yaxis": "y2",
        "marker": {"color": "#ccc"},
        "hoverinfo": "none",
        "legendgroup": "Bollinger Bands"
    }
    bband_down = {
        "line": {"width": 3},
        "mode": "lines",
        "type": "scatter",
        "x": df['Date'],
        "y": df['lower_bb'],
        "yaxis": "y2",
        "marker": {"color": "#ccc"},
        "hoverinfo": "none",
        "showlegend": False,
        "legendgroup": "Bollinger Bands"
    }
    layout = go.Layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "showticklabels": False, "showgrid": False,
               "rangebreaks": [
                   dict(bounds=["sat", "mon"]),  # hide weekends
                   # https://plotly.com/python/time-series/#hiding-weekends-and-holidays
               ]},
        yaxis={"visible": False, "showticklabels": False, "showgrid": False}
    )
    fig = go.Figure(data=[candle, moving_avg, bband_up, bband_down], layout=layout)
    fig.update_layout(showlegend=False)
    fig.show()


if __name__ == '__main__':
    candle_charts('BA', '2015-01-01', '2015-01-15')
