import pandas as pd
import numpy as np
from datetime import datetime
import pandas_market_calendars as mcal
import os
import re
import warnings
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LinearRegression
pd.options.mode.chained_assignment = None
from yahoo_fin import stock_info as si
import financial_metrics as fm
from plotly.subplots import make_subplots

from psycopg2 import connect

def performance_chart(tr, BM):
    '''
    :param tr: track record dataframe
    :param BM: the ticker for a desired benchmark ETF
    '''
    tr = tr.drop_duplicates()
    if 'Date' in tr.columns:
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    else:
        tr = tr.reset_index()
        s_date = tr['Date'].iloc[0]
    e_date = tr['Date'].iloc[-1]
    spy = si.get_data(BM.upper(), start_date = s_date, end_date = e_date)['close']
    #tr = port.track_record.drop_duplicates()
    tr['Date'] = pd.to_datetime(tr['Date'])
    plot_dat = tr.set_index('Date').join(spy)
    plot_dat = plot_dat.filter(['Value', 'close']).dropna()
    plot_dat = plot_dat.rename(columns = {"Value": "Portfolio", "close":'{}'.format(BM)})
    plot_dat['Port_Performance'] = (plot_dat['Portfolio']/plot_dat['Portfolio'].iloc[0])-1
    plot_dat['{} Performance'.format(BM)] = (plot_dat['{}'.format(BM)]/plot_dat['{}'.format(BM)].iloc[0])-1
    color_codes = ["#FFCB05", "#00274C", "#9A3324", "#D86018", "#75988d", "#A5A508", "#00B2A9", "#2F65A7", "#702082"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = plot_dat.index, y = plot_dat['Port_Performance'], name = 'Model', mode = 'lines', line = dict(color = color_codes[0])))
    fig.add_trace(go.Scatter(x = plot_dat.index, y = plot_dat['{} Performance'.format(BM)], name = '{}'.format(BM), mode = 'lines', line = dict(color = color_codes[1])))
    fig.update_layout(#width = 950, height = 500,
                            margin=dict(l=20, r=20, t=50, b=10),
                            paper_bgcolor='white',
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis_title="Return on Investment (ROI)",
                            legend=dict( orientation="h"),
                            yaxis_tickformat = '.0%',
                            title= dict(text='Performance Chart', font = dict(size = 20, color = 'white'), x = 0.5, y = 0.96))
    fig.update_yaxes(range=[-.3, .8])

    return fig


def risk_adjusted_metrics(tr, BM):
    '''
    :param tr: track record dataframe
    :param BM: the ticker for a desired benchmark ETF
    '''
    color_codes = ["#FFCB05", "#00274C", "#9A3324", "#D86018", "#75988d", "#A5A508", "#00B2A9", "#2F65A7", "#702082"]
    if 'Date' in tr.columns:
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    else:
        tr = tr.reset_index()
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    spy = si.get_data(BM, start_date = s_date, end_date = e_date)['close']
    tr = tr.drop_duplicates()
    tr['Date'] = pd.to_datetime(tr['Date'])
    plot_dat = tr.set_index('Date').join(spy)
    plot_dat = plot_dat.filter(['Value', 'close']).dropna()
    plot_dat = plot_dat.rename(columns = {"Value": "Portfolio", "close":'{}'.format(BM)})
    plot_dat['Port_daily'] = plot_dat['Portfolio'].pct_change().fillna(0)
    plot_dat['{}_daily'.format(BM)] = plot_dat['{}'.format(BM)].pct_change().fillna(0)
    port_sharpe = fm.sharpe_ratio(list(plot_dat['Port_daily']))
    spy_sharpe = fm.sharpe_ratio(list(plot_dat['{}_daily'.format(BM)]))
    port_sort = fm.sortino_ratio(list(plot_dat['Port_daily']))
    spy_sort = fm.sortino_ratio(list(plot_dat['{}_daily'.format(BM)]))
    port_trey = fm.treynor(list(plot_dat['Port_daily']), list(plot_dat['{}_daily'.format(BM)]), rf=0.0015)
    spy_trey = fm.treynor(list(plot_dat['{}_daily'.format(BM)]), list(plot_dat['{}_daily'.format(BM)]), rf=0.0015)
    port_dd = fm.max_drawdow(list(plot_dat['Port_daily']))
    spy_dd = fm.max_drawdow(list(plot_dat['{}_daily'.format(BM)]))
    port_calm = fm.calmer(list(plot_dat['Port_daily']))
    spy_calm = fm.calmer(list(plot_dat['{}_daily'.format(BM)]))
    metrics = ['Sharpe', 'Sortino', 'Treynor', 'Max Drawdown', 'Calmer']
    port = [port_sharpe, port_sort, port_trey, port_dd, port_calm ]
    sp = [spy_sharpe, spy_sort, spy_trey, spy_dd, spy_calm]
    fig_2 = go.Figure(data = [
        go.Bar(name = 'Model', x=metrics, y=port, text=['{:.2f}'.format(i) for i in port], marker_color=color_codes[2]),
        go.Bar(name = '{}_daily'.format(BM), x=metrics, y=sp, text=['{:.2f}'.format(i) for i in sp], marker_color=color_codes[4]),
    ])

    fig_2.update_layout( #width = 950, height = 400,
                        barmode='group',
                        margin=dict(l=20, r=20, t=50, b=10),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        legend=dict( orientation="h"),
                        yaxis_tickformat = '.0f',
                        yaxis_title="Ratio",
                        title= dict(text='Risk Adjusted Metrics', font = dict(size = 20, color = 'white'), x = 0.5, y = 0.96))
    fig_2.update_yaxes(range=[-3, 20])

    return fig_2


def risk_to_ret(tr, BM):
    '''
    :param tr: track record dataframe
    :param BM: the ticker for a desired benchmark ETF
    '''
    color_codes = ["#FFCB05", "#00274C", "#9A3324", "#D86018", "#75988d", "#A5A508", "#00B2A9", "#2F65A7", "#702082"]
    if 'Date' in tr.columns:
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    else:
        tr = tr.reset_index()
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    spy = si.get_data(BM, start_date = s_date, end_date = e_date)['close']
    tr = tr.drop_duplicates()
    tr['Date'] = pd.to_datetime(tr['Date'])
    plot_dat = tr.set_index('Date').join(spy)
    plot_dat = plot_dat.filter(['Value', 'close']).dropna()
    plot_dat = plot_dat.rename(columns = {"Value": "Portfolio", "close":'{}'.format(BM)})
    plot_dat['Port_daily'] = plot_dat['Portfolio'].pct_change().fillna(0)
    plot_dat['{}_daily'.format(BM)] = plot_dat['{}'.format(BM)].pct_change().fillna(0)
    port_sharpe = fm.sharpe_ratio(list(plot_dat['Port_daily']))
    spy_sharpe = fm.sharpe_ratio(list(plot_dat['{}_daily'.format(BM)]))
    x = [np.std(list(plot_dat['Port_daily']))*np.sqrt(365), np.std(list(plot_dat['{}_daily'.format(BM)]))*np.sqrt(365)]
    y = [np.mean(list(plot_dat['Port_daily']))*365, np.mean(list(plot_dat['{}_daily'.format(BM)]))*365]
    z = [port_sharpe, spy_sharpe]
    names = ['Model', '{}'.format(BM)]
    fig_3 = go.Figure()
    fig_3.add_trace(go.Scatter(
        x = np.array(x[0]), y = np.array(y[0]),
        mode  = 'markers',
        marker = dict(color = color_codes[0],
                      size = (z[0]*2)**2), name = "Model"))
    fig_3.add_trace(go.Scatter(
        x = np.array(x[1]), y = np.array(y[1]),
        mode  = 'markers',
        marker = dict(color = color_codes[1],
                      size = (z[1]*2)**2), name = '{}_daily'.format(BM)))
    fig_3.update_layout(#width = 700, height = 400,
                            margin=dict(l=20, r=20, t=35, b=10),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            legend=dict(orientation="v",yanchor="top",y=2),
                            yaxis_tickformat = '.0%',
                            xaxis_tickformat = '.0%',
                            legend_title_text=('Size = Sharpe Ratio'),
                            xaxis = dict(title =  'Annualized Volatility'),
                            yaxis = dict(title =  'Annualized Return'),
                            title= dict(text='Risk vs Return', font = dict(size = 20, color = 'white'), x = 0.55, y = 0.96))
    fig_3.update_yaxes(range=[0, 2])
    fig_3.update_xaxes(range=[0, .5])

    return fig_3

def sector_plot(snap_port, snap_cash, date):
    '''
    :param snap_port:
    :param snap_cash:
    :param date: the date of the desired allocation breakdown
    '''
    color_codes = ["#FFCB05", "#00274C", "#9A3324", "#D86018", "#75988d", "#A5A508", "#00B2A9", "#2F65A7", "#702082"]
    sectors = pd.read_csv("assets/fundamentals/ms_sectors.csv", index_col = 0)
    sectors = sectors.rename(columns = {"ticker": "Ticker"})
    snap_port = snap_port.merge(sectors, on = 'Ticker', how= 'inner')
    snap_port = snap_port.groupby("sector").sum().filter(["Current Value"])
    snap_port['% of Portfolio'] = snap_port["Current Value"]/(snap_port["Current Value"].sum() + snap_cash)
    gics = "Cash"
    pr_of_port = 1-snap_port["% of Portfolio"].sum()
    curv = snap_cash
    cash_pos = pd.DataFrame({"Current Value": curv, "% of Portfolio": pr_of_port}, index = [gics])
    snap = pd.concat([cash_pos, snap_port])
    fig_4 = go.Figure(data = [go.Pie(labels = snap.index, values = snap['% of Portfolio'], hole = 0.3)])
    fig_4.update_traces(hoverinfo='label+percent', textinfo='label + percent', textfont_size=10,textposition='inside',
                          marker=dict(colors=color_codes, line=dict(color='#000000', width=2)))
    fig_4.update_layout(showlegend=False,
                            # width = 475, height = 500,
                            margin=dict(l=10, r=10, t=35, b=5),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            title= dict(text='Portfolio Allocation'.format(date), font = dict(size = 20, color = 'white'), x = 0.5, y = 0.98))
    return fig_4


def capm_res(tr, BM):
    '''
    :param tr: track record dataframe
    :param BM: the ticker for a desired benchmark ETF
    '''
    color_codes = ["#FFCB05", "#00274C", "#9A3324", "#D86018", "#75988d", "#A5A508", "#00B2A9", "#2F65A7", "#702082"]
    if 'Date' in tr.columns:
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    else:
        tr = tr.reset_index()
        s_date = tr['Date'].iloc[0]
        e_date = tr['Date'].iloc[-1]
    spy = si.get_data(BM, start_date = s_date, end_date = e_date)['close']
    tr = tr.drop_duplicates()
    tr['Date'] = pd.to_datetime(tr['Date'])
    plot_dat = tr.set_index('Date').join(spy)
    plot_dat = plot_dat.filter(['Value', 'close']).dropna()
    plot_dat = plot_dat.rename(columns = {"Value": "Portfolio", "close":'{}'.format(BM)})
    plot_dat['Port_daily'] = plot_dat['Portfolio'].pct_change().fillna(0)
    plot_dat['{}_daily'.format(BM)] = plot_dat['{}'.format(BM)].pct_change().fillna(0)
    X = plot_dat['{}_daily'.format(BM)][:, None]
    y = plot_dat['Port_daily']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)
    model = LinearRegression()
    model.fit(X_train , y_train)
    x_range = np.linspace(X.min(), X.max(), 500)
    y_range = model.predict(x_range.reshape(-1, 1))
    score = model.score(X_train , y_train)
    alpha = model.intercept_*365
    beta = model.coef_[0]
    bm_for_capm = si.get_data(BM)['close'].pct_change().dropna().mean()*365
    aer = (beta * bm_for_capm) + alpha
    correl = np.sqrt(score)
    X_bars = ['R Squared', "Correlation", "Beta", "Alpha", "Expected Return"]
    y_bars = [score, correl, beta, alpha, aer]
    bar_text = ['{:.2%}'.format(score), '{:.2f}'.format(correl), '{:.2f}'.format(beta),'{:.2%}'.format(alpha), '{:.2%}'.format(aer)]
    fig = make_subplots(rows=1, cols=3,
                       specs = [[{'colspan':2,},None,{'type':'bar'}]],
                       subplot_titles = ('Regression: Portfolio vs {}'.format(BM), "CAPM Metrics"), vertical_spacing=0.01)
    fig.add_trace(
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers', marker = {'color':color_codes[0]}), row = 1, col = 1),
    fig.add_trace(
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers',  marker = {'color':color_codes[1]}), row = 1, col = 1),
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, name='prediction', line = {'color':color_codes[2]}), row = 1, col = 1)
    fig.add_trace(
        go.Bar(y = X_bars, x=y_bars, text = bar_text, name = 'CAPM Stats', marker = {'color':color_codes[4]}, orientation = 'h'), row = 1, col=3)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(tickfont = {'size':8}, row=1, col=3)
    fig.update_layout(width = 1000, height = 500,
                      xaxis_tickformat = '.2%',
                      yaxis_tickformat = '.2%',
                            margin=dict(l=10, r=10, t=60, b=5),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            title= dict(text='Capital Asset Pricing Model', font = dict(size = 20, color = 'black'), x = 0.5, y = 0.98))

    return fig

def fundamental_chart(metric, stock_sector, date = '2021-06-30'):
    '''function to call fundamental charts onto dashboard
    :param metric: (string) p_e, p_b, ev_sales, ev_ebitda
    :param stock_sector: (string) a stock ticker or the name of a GICS sector
    :param date: (optional - string) a reference date must be in YYYY-MM-DD format
    '''

    sectors = pd.read_csv("assets/fundamentals/ms_sectors.csv", index_col = 0)
    sectors = sectors.rename(columns = {"ticker": "Ticker"})

    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = "SELECT * FROM raw_data WHERE date = '{}'".format(date)
    data = pd.read_sql_query(query,conn).filter(['Instrument','3m_avg_mkt_cap', '{}'.format(metric)]).rename(columns = {"Instrument": "Ticker"})
    data = data.merge(sectors, on = 'Ticker')
    sector_list = sectors['sector'].unique()
    colors = ['#4e5d6c','#191970']
    if stock_sector in sector_list:
        plot_df = data[data['sector']==stock_sector]
        if len(plot_df)>10:
            plot_df = plot_df.sort_values(['3m_avg_mkt_cap']).iloc[0:10]
        else:
            plot_df = plot_df.sort_values(['3m_avg_mkt_cap'])
        plot_df['color'] = [colors[0] for i in range(0, len(plot_df))]

    else:
        stk = data[data['Ticker']==stock_sector.upper()]['sector'].iloc[0]
        plot_df = data[data['sector']==stk]
        stock_line = plot_df[plot_df['Ticker']==stock_sector.upper()]
        if len(plot_df)>10:
            plot_df = plot_df[plot_df['Ticker']!=stock_sector.upper()].sort_values(['3m_avg_mkt_cap'], ascending = False).iloc[0:9]
            plot_df['color'] = [colors[0] for i in range(0, len(plot_df))]
            stock_line['color'] = colors[1]
            plot_df = pd.concat([plot_df, stock_line])
        else:
            plot_df = plot_df[plot_df['Ticker']!=stock_sector.upper()].sort_values(['3m_avg_mkt_cap'], ascending = False)
            plot_df['color'] = [colors[0] for i in range(0, len(plot_df))]
            stock_line['color'] = color[1]
            plot_df = pd.concat([plot_df, stock_line])
    plot_df = plot_df.sort_values(['{}'.format(metric)], ascending = False)

    metric_title = metric.replace('_','/').upper()
    fig = go.Figure(go.Bar(
        y = plot_df['Ticker'],
        x = plot_df['{}'.format(metric)],
        text = ['{:.2f}x'.format(i) for i in list(plot_df['{}'.format(metric)])],
        textfont = {'size':8},
        textposition="inside",
        marker_color = plot_df['color'],
        orientation = 'h'))

    fig.update_xaxes(showticklabels=False,showgrid=False)
    fig.update_yaxes(tickfont = {'size':8})
    fig.update_layout(#width = 400, height = 400,
                      xaxis_tickformat = '.2f',
                            margin=dict(l=10, r=10, t=20, b=5),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            font=dict(color='white'),
                            title= dict(text='{} Ratio Comparison for <br> {}'.format(metric_title, stock_sector), font = dict(size = 8, color = 'white'), x = 0.5, y = 0.98),
                            )
    return fig

def mlp_stat_chart(csv_path, stat = 'loss'):
    '''
    plots training and validation plots for MF_MLP model
    :param csv_path: (str) path to csv file containing epoch results from MF_MLP model
    :param stat: (str) one of the metrics used to evaluate the model (loss, binary_accuracy, precision, or recall)
    '''
    color_list = ['#6929c4', '#1192e8', '#005d5d', '#9f1853', '#fa4d56', '#570408', '#198038', '#002d9c', '#ee538b', '#b28600', '#009d9a', '#012749']
    df = pd.read_csv(csv_path)
    filts = [i for i in df.columns if stat in i]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = [i for i in range(0, len(df))],
                                 y = list(df[filts[0]]),
                                 name = 'Train',
                                 line = {'color':'#1192e8'}))
    # fig.add_trace(go.Scatter(x = [i for i in range(0, len(df))],
    #                              y = list(df[filts[1]]),
    #                              name = 'Train',
    #                              mode = 'lines+markers',
    #                              marker = {'color':'#002d9c'}))

    # Adding IF statement to handle Jeff and Joey's different vizzes
    if len(filts) > 1:
        fig.add_trace(go.Scatter(x = [i for i in range(0, len(df))],
                                     y = list(df[filts[1]]),
                                     name = 'Train',
                                     mode = 'lines+markers',
                                     marker = {'color':'#002D9C'}))


    fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor= 'white',
                margin=dict(l=0, r=0, t=15, b=5),
                # height = 250,
                # width = 500,
                font=dict(color='white',size=10),
                showlegend=False,
                title = dict(text = 'Multifactor MLP {} Results'.format(stat), x = 0.5, y = 0.97, font = {'size': 10,'color':'white'})
                )
    fig.update_layout(hovermode="x unified")
    return fig
