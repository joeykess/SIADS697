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
                    # Adding drop down to filter by ticker
                    dcc.Dropdown(id='ticker',
                        options=[{'label': i, 'value': i} for i in list(stock_df['ticker'].unique())],
                        value='CSCO') # the default is code_module AAA
                        ]),

                    # Line two: portoflio and ticker info
                    html.Div([
                        html.H2('Portfolio Performance',style=portfolio_style),
                        dcc.Graph(id='price_chart',style=chart_style),
                        html.H2(id='news_list',style=news_style_b)
                        # html.Div(id='news_list',children=news_info,style=news_style_b)
                        ]),

                    # Line three: other info, notyet defined
                    html.Div([
                        html.H2('Other Portfolio Statistics',style=portfolio_style_b),
                        html.H2('News Info',style=chart_style_b),
                        ],style={'height':200})
                    ])

# Callback to connect input(s) to output(s) for Tab 1
@app.callback(dash.dependencies.Output('price_chart','figure'),
    [dash.dependencies.Input('ticker','value')])

# Step 3: Define the graph with plotly express
def update_ticker(ticker):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=stock_df[stock_df['ticker']==ticker]['Date'],
                             y=stock_df[stock_df['ticker']==ticker]['Close'],
                            line={"color": "#228B22"},
                            mode="lines"))

    fig.update_layout(title_text=f'{ticker} Closing Price',title_x=0.5,
                         template="plotly_dark",font=dict(size=10),xaxis_showgrid=False,
                         yaxis_title="Closing Price",margin={"r": 20, "t": 35, "l": 20, "b": 10})

    return fig

# Creating callback to get news when ticker changes
@app.callback(dash.dependencies.Output('news_list', 'children'),
[dash.dependencies.Input('ticker', 'value')])
def update_news(ticker):

    seeking_alpha = fn.SeekingAlpha(topics=['$'+ticker], save_feeds=True)

    news = seeking_alpha.get_news()

#     title_strings = [re.sub('[^A-Za-z0-9,\s]+', '', item['title']).lower().replace(" ", "-") for item in news[:5]]
#     url_ids = [re.sub('MarketCurrent:','news/',item) for item in news[:5]]

    return html.Div([html.H2(f'News for {ticker}',style={'backgroundColor':'gray','color':'white'}),\
                dbc.ListGroup(
                    [dbc.ListGroupItem(
                        [html.Div([
                            html.A(html.P(item['title'],style=news_style),\
                            href=(item['link'])),\
                            html.A(html.P(item['published'],style=news_style))
                            ])
                        ],color='gray') for item in news]
                ,flush=True)
            ])
