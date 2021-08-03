import os
import time
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import FinNews as fn
import re
import pathlib

# Dash modules
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from jupyter_dash import JupyterDash
import plotly.express as px
import plotly.graph_objects as go

from apps.ind_css import *
from app import app

# Not using separate callback files
# from apps.portfolio_performance_cbs import *

# Getting all file paths

import pathlib

PATH = pathlib.Path(__file__).parent

DATA_PATH = str(PATH.joinpath('../assets/historical-symbols'))

# path = r'../assets/historical-symbols' # use your path
all_files = glob.glob(DATA_PATH + "/*.csv")

# Creating list to append all ticker dfs to
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

# Concat all ticker dfs
stock_df = pd.concat(li, axis=0, ignore_index=True,sort=True)

stock_df['Date'] = pd.to_datetime(stock_df['Date'])

sector_df = stock_df.groupby(['sector','Date']).mean()['Close'].reset_index()

layout = html.Div([

                html.Div([
                    html.Div([
                        # Adding drop down to filter by ticker
                        html.A('Pick How You Want to  Analyze Data:'),
                        dcc.Dropdown(id='ticker_filter',
                            options=[{'label': i, 'value': i} for i in ['Ticker','Sector']],
                            value='Ticker'), # the default is code_module AAA

                        # dcc.Dropdown(id='industry_ticker',
                        #     options=[{'label': i, 'value': i} for i in list(stock_df['sector'].unique())],
                        #     value='Technology',style={'margin':'5px','display':'inline-block'}) # the default is code_module AAA
                            ],style={'margin':'5px','width':'30%','border':'thin lightgrey solid','display':'inline-block'}),

                    html.Div([
                        # Adding drop down to filter by ticker
                        html.A('Filter Data:'),
                        dcc.Dropdown(id='data_filter',
                            options=[{'label': '', 'value': ''}],
                            value='CSCO'), # the default is code_module AAA
                            ],style={'margin':'5px','width':'30%','border':'thin lightgrey solid','display':'inline-block'}),

                    # Adding date filter buttons for charts
                    html.Div([

                        # HTM Div for Buttons
                        html.Div([
                            html.Button('7 Days', id='btn-nclicks-1',n_clicks=0,style={'width':'23.7%','margin':'2px'}),
                            html.Button('30 Days', id='btn-nclicks-2',n_clicks=0,style={'width':'23.7%','margin':'2px'}),
                            html.Button('1 Year', id='btn-nclicks-3',n_clicks=0,style={'width':'23.7%','margin':'2px'}),
                            html.Button('All', id='btn-nclicks-4',n_clicks=0,style={'width':'23.7%','margin':'2px'})]),
                            html.Div([dcc.Dropdown(id='MA_filter',
                                        options=[{'label': i, 'value': i} for i in ['60 Day MA','200 Day MA']],
                                        multi=True)])
                            ],style={'margin':'5px','width':'35%','border':'thin lightgrey solid','display':'inline-block','float':'right'})
                        ],style={'margin':'5px','width':'99%','border':'thin lightgrey solid'}),

                    # Add dropdown for category (stock name, industry, etc) and do conditional formatting for second dropdown

                    # Line two: portoflio and ticker info
                    html.Div([
                        html.H2('Portfolio Performance',style=portfolio_style),
                        dcc.Graph(id='chart-1',style=chart_style),
                        html.Div(id='news_list',style=news_style_b)
                        # html.Div(id='news_list',children=news_info,style=news_style_b)
                        ]),

                    # Line three: other info, notyet defined
                    html.Div([
                        html.H2('Sector Mix (Pie Chart?)',style=portfolio_style),
                        dcc.Graph(id='chart-2',style=chart_style),
                        html.H2('Twitter Sentiment',style=news_style_b)
                        ])
                    ])

# Callback to connect input(s) to output(s) for Tab 1
@app.callback(dash.dependencies.Output('chart-1','figure'),
    [dash.dependencies.Input('ticker_filter','value'),
    dash.dependencies.Input('data_filter','value'),
    dash.dependencies.Input('btn-nclicks-1', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-2', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-3', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-4', 'n_clicks'),
    dash.dependencies.Input('MA_filter', 'value')])

# Step 3: Define the graph with plotly express
def update_ticker(ticker_filter,ticker,btn1,btn2,btn3,btn4,ma_filters):

    from datetime import datetime, timedelta

    fig = go.Figure()

    if ticker_filter == 'Ticker':

        df = stock_df[stock_df['ticker']==ticker]
        df = df.set_index('Date')

    else:
        df = sector_df[sector_df['sector']==ticker]
        df = df.set_index('Date')

    # Adding 60 Day Moving Average
    df['60 Day MA'] = df.Close.rolling(window=60).mean()
    df['200 Day MA'] = df.Close.rolling(window=200).mean()

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'btn-nclicks-1' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=7)]
    elif 'btn-nclicks-2' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=30)]
    elif 'btn-nclicks-3' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=365)]
    elif 'btn-nclicks-4' in changed_id:
        tick_df = df
    else:
        tick_df = df

    fig.add_trace(go.Scatter(x=tick_df.index,
                         y=tick_df['Close'],
                        line={"color": "#228B22"},
                        mode="lines",
                        name='Closing Price'))

    fig.update_layout(title_text=f'{ticker} Closing Price',title_x=0.5,
                         template="ggplot2",font=dict(size=10,color='white'),xaxis_showgrid=False,
                         paper_bgcolor='rgba(0,0,0,0)',
                         yaxis_title="Closing Price",margin={"r": 20, "t": 35, "l": 20, "b": 10},
                         showlegend=False)

    try:
        if '60 Day MA' in ma_filters:
            fig.add_trace(go.Scatter(x=tick_df.index,
                                 y=tick_df['60 Day MA'],
                                line={"color": "gray","width":1},
                                mode="lines",
                                name='60 Day Moving Avg'))
    except:
        pass

    try:
        if '200 Day MA' in ma_filters:
            fig.add_trace(go.Scatter(x=tick_df.index,
                                 y=tick_df['200 Day MA'],
                                line={"color": "black","width":1},
                                mode="lines",
                                name='200 Day Moving Avg'))
    except:
        pass

    fig.update_layout(hovermode="x unified")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=10))

    return fig

# Creating callback to get news when ticker changes
@app.callback(dash.dependencies.Output('news_list', 'children'),
[dash.dependencies.Input('ticker_filter','value'),
 dash.dependencies.Input('data_filter', 'value')])
def update_news(ticker_filter,ticker):

    if ticker_filter == 'Sector':
        topics = ticker.lower().split(" ")
        seeking_alpha = fn.SeekingAlpha(topics=topics)
        news = seeking_alpha.get_news()

    else:
        seeking_alpha = fn.SeekingAlpha(topics=['$'+ticker])
        news = seeking_alpha.get_news()

    return html.Div([html.H2(f'News for {ticker}',style={'backgroundColor':'gray','color':'white','fontSize':14,'border-bottom':'3px solid white'}),\
                dbc.ListGroup(
                    [dbc.ListGroupItem(
                        [html.Div([
                            html.A(html.P(item['title'],style=news_style),\
                            href=(item['link']),target="_blank"),\
                            html.A(html.P(item['published'],style=news_style_c))
                            ])
                        ],color='gray') for item in news]
                ,flush=True)
            ])

# Creating callback to get conditially set options in dropdown filter
@app.callback(
    dash.dependencies.Output('data_filter', 'options'),
    [dash.dependencies.Input('ticker_filter', 'value')])
def update_dropdown(filter_option):
    if filter_option == 'Ticker':
        col_labels = [{'label' :k, 'value' :k} for k in list(stock_df['ticker'].unique())]
        return col_labels
    elif filter_option == 'Sector':
        col_labels = [{'label' :k, 'value' :k} for k in list(stock_df['sector'].unique())]
        return col_labels

# Callback to connect input(s) to output(s) for Tab 1
@app.callback(dash.dependencies.Output('chart-2','figure'),
    [dash.dependencies.Input('data_filter','value'),
    dash.dependencies.Input('btn-nclicks-1', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-2', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-3', 'n_clicks'),
    dash.dependencies.Input('btn-nclicks-4', 'n_clicks')])

# Step 3: Define the graph with plotly express
def update_ticker(ticker,btn1,btn2,btn3,btn4):

    from datetime import datetime, timedelta

    fig = go.Figure()

    df = stock_df[stock_df['ticker']==ticker]
    df = df.set_index('Date')

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'btn-nclicks-1' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=7)]
    elif 'btn-nclicks-2' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=30)]
    elif 'btn-nclicks-3' in changed_id:
        tick_df = df[df.index >= df.index.max()-timedelta(days=365)]
    elif 'btn-nclicks-4' in changed_id:
        tick_df = df
    else:
        tick_df = df

    fig.add_trace(go.Candlestick(x=tick_df.index,
                    open=tick_df['Open'],
                    high=tick_df['High'],
                    low=tick_df['Low'],
                    close=tick_df['Close']))

    fig.update_layout(title_text=f'{ticker} Candlestick Chart',title_x=0.5,
                         template="ggplot2",font=dict(size=10,color='white'),xaxis_showgrid=False,
                         paper_bgcolor='rgba(0,0,0,0)',
                         yaxis_title="Closing Price",margin={"r": 20, "t": 35, "l": 20, "b": 10},
                         xaxis_rangeslider_visible=False)

    return fig
