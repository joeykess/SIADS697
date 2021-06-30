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
                            ],style={'margin':'5px','width':'30%','border':'thin lightgrey solid','display':'inline-block'})
                        # dcc.Dropdown(id='industry_ticker',
                        #     options=[{'label': i, 'value': i} for i in list(stock_df['sector'].unique())],
                        #     value='Technology',style={'margin':'5px','display':'inline-block'}) # the default is code_module AAA
                            ],style={'margin':'5px','width':'99%','border':'thin lightgrey solid'}),

                    # Add dropdown for category (stock name, industry, etc) and do conditional formatting for second dropdown

                    # Line two: portoflio and ticker info
                    html.Div([
                        html.H2('Portfolio Performance',style=portfolio_style),
                        dcc.Graph(id='price_chart',style=chart_style),
                        html.Div(id='news_list',style=news_style_b)
                        # html.Div(id='news_list',children=news_info,style=news_style_b)
                        ]),

                    # Line three: other info, notyet defined
                    html.Div([
                        html.H2('Other Portfolio Statistics',style=portfolio_style),
                        html.H2('Stock in Sector P/Es?',style=chart_style),
                        ])
                    ])

# Callback to connect input(s) to output(s) for Tab 1
@app.callback(dash.dependencies.Output('price_chart','figure'),
    [dash.dependencies.Input('data_filter','value')])

# Step 3: Define the graph with plotly express
def update_ticker(ticker):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=stock_df[stock_df['ticker']==ticker]['Date'],
                             y=stock_df[stock_df['ticker']==ticker]['Close'],
                            line={"color": "#228B22"},
                            mode="lines"))

    fig.update_layout(title_text=f'{ticker} Closing Price',title_x=0.5,
                         template="ggplot2",font=dict(size=10,color='white'),xaxis_showgrid=False,
                         paper_bgcolor='rgba(0,0,0,0)',
                         yaxis_title="Closing Price",margin={"r": 20, "t": 35, "l": 20, "b": 10})

    return fig

# Creating callback to get news when ticker changes
@app.callback(dash.dependencies.Output('news_list', 'children'),
[dash.dependencies.Input('data_filter', 'value')])
def update_news(ticker):

    seeking_alpha = fn.SeekingAlpha(topics=['$'+ticker], save_feeds=True)

    news = seeking_alpha.get_news()

#     title_strings = [re.sub('[^A-Za-z0-9,\s]+', '', item['title']).lower().replace(" ", "-") for item in news[:5]]
#     url_ids = [re.sub('MarketCurrent:','news/',item) for item in news[:5]]

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
