import yahoo_fin.stock_info as yfi
import cufflinks as cf
import plotly.io as pio


def tech_chart(stock, start_date=str(), end_date=str()):
    df = yfi.get_data(stock, start_date=start_date, end_date=end_date, interval='1d')
    fig = cf.QuantFig(df, legend='top', name='{}'.format(stock),
                      rangeselector=dict(steps=['Reset', '1Y', '6M', '3M', '1M'],
                                         bgcolor=('rgb(150, 200, 250)', .1),
                                         fontsize=12, fontfamily='monospace', x=0, y=1), rangeslider=True,
                      hoverformat='y')
    fig.add_volume(colorchange=True, dimensions=(1200, 100), name='Volume')
    fig.add_macd(fast_period=12, slow_period=26, signal_period=9, name='MACD', )
    fig.studies['macd']['display'].update(legendgroup=True)
    fig.data.update(showlegend=False)
    fig.add_ema(colors='brown', name='EMA')
    fig.add_bollinger_bands(periods=20, boll_std=2, colors=['magenta', 'grey'], name='BOLL', showlegend=False)
    fig.add_rsi(periods=20, name='RSI', legendgroup=False, showbands=False)
    fig.theme.update(theme='ggplot')
    fig['data']['hoverinfo'] = 'all'
    fig = fig.figure(asImage=True, title={
        'text': "{}".format(stock), 'x': 0.5}, dimensions=(1200, 1500), spacing=0.03)

    return pio.write_html(fig, file='{}_technicals.html'.format(stock), auto_open=True)


if __name__ == '__main__':
    tech_chart("SPY", "2005-12-31", "2021-06-28")
