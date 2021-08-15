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
from dash_table import DataTable, FormatTemplate
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from jupyter_dash import JupyterDash
import plotly.express as px
import plotly.graph_objects as go

from apps.ind_css import *
from app import app

from psycopg2 import connect

# Not using separate callback files
# from apps.portfolio_performance_cbs import *


def import_technical_features():
    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = 'SELECT "Date","sector","ticker","Close","Open","High","Low" FROM technical_features_daily'
    data = pd.read_sql_query(query,conn)
    data = data.sort_values(['ticker', 'Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data
stock_df = import_technical_features()

def import_open_positions():
    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = "SELECT * FROM open_positions"
    data = pd.read_sql_query(query,conn)
    data = data.sort_values(['key', 'model'])
    return data
open_pos_df = import_open_positions().drop('index',axis=1)


sector_df = stock_df.reset_index().groupby(['sector','Date']).mean()['Close'].reset_index()
sentiment_df = pd.read_csv('assets/models/tyler_rf_daily_update/sentiment_analysis.csv')

model_dict = {'Random Forest Regressor 120/30': 'RF Reg_target_120_rebal_30_2017-01-01',
              'Random Forest Regressor 120/60': 'RF Reg_target_120_rebal_60_2017-01-01',
              'Random Forest Regressor 60/30': 'RF Reg_target_60_rebal_30_2017-01-01',
              'Random Forest Regressor 7/7': 'RF Reg_target_7_rebal_7_2017-01-01',
              'Multi Factor Multi-Layer Preceptron': 'MF_MLP'
              # 'CNN Visual Pattern Recognition': '75percent_confidence_no_holding_15m_cnn'
             }

layout = html.Div([

                html.Div([
                    html.Div([
                        # Adding drop down to filter by ticker
                        html.A('Pick How You Want to  Analyze Data:'),
                        dcc.Dropdown(id='ticker_filter',
                            options=[{'label': i, 'value': i} for i in ['Ticker','Sector']],
                            value='Ticker',clearable=False), # the default is code_module AAA

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
                        html.Div([
                            dcc.Graph(id='indicator-graph',style={'height':'25%','float':'top'}),
                            html.Div(id='portfolio-table')
                        ],style=portfolio_style),
                        dcc.Graph(id='chart-1',style=chart_style),
                        html.Div(id='news_list',style=news_style_b)
                        # html.Div(id='news_list',children=news_info,style=news_style_b)
                        ]),

                    # Line three: other info, notyet defined
                    html.Div([
                        html.H2('Placeholder',style=portfolio_style),
                        dcc.Graph(id='chart-2',style=chart_style),
                        html.Div(id='sentiment',style=sentiment_style)
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
        # df = df.set_index('Date')

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

    fig.update_layout(title=dict(text=f'{ticker} Closing Price',font = dict(size = 20, color = 'white'), x = 0.5, y = 0.96),
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

    return html.Div([html.H2(f'News for {ticker}',style={'color':'white','fontSize':14,'border-bottom':'3px solid white'}),\
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
    # df = df.set_index('Date')

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

# Creating callback for twitter sentiment
@app.callback(dash.dependencies.Output('sentiment','children'),
    [dash.dependencies.Input('data_filter','value')])
# Step 3: Define the graph with plotly express
def update_sentiment(ticker_filter):

    if ticker_filter == 'Sector':
        sector = ticker
    else:
        sector = stock_df[stock_df['ticker']==ticker_filter]['sector'].unique()[0]

    df = sentiment_df[sentiment_df['sector']==sector]\
                    .nlargest(10,'Mentions')[['Ticker','Sentiment','Trend']]

    table = DataTable(
        id='sentiment_table',
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell=dict(textAlign='center'),
        style_header=dict(backgroundColor="#191970",color='white'),
        style_data=dict(backgroundColor="gray",color='black'),

        # Setting conditional styles for sentiment and trends
         style_data_conditional=[
            {'if': {'filter_query': '{Sentiment} = "good" || {Sentiment} = "very good"','column_id': 'Sentiment'},
                'color': '#03CD1E'},
            {'if': {'filter_query': '{Trend} = "down"','column_id': 'Trend'},
                'color': '#FF0000'},
            {'if': {'filter_query': '{Trend} = "up"','column_id': 'Trend'},
                'color': '#03CD1E'}],

        style_as_list_view=True
    )

    return html.Div([html.A(f'Top Sentiment for Sector: {sector}',style={'color':'white','fontsize':8}),table])

@app.callback(dash.dependencies.Output('indicator-graph', 'figure'),
              [dash.dependencies.Input('data_filter','value'),
               dash.dependencies.Input('model_filter','value')])
def update_port_value(value,model):

    # For testing portfolio
    mod_filter = model_dict[model]

    open_pos_df_chart = open_pos_df[open_pos_df['model']==mod_filter]
    # Getting max date for input, may make configurable later
    date = open_pos_df_chart['key'].values[-1][-10:]

    # date_filter = '2020-09-04'
    date_filter = date
    test_df = open_pos_df[(open_pos_df['model']==mod_filter)&\
                          (open_pos_df['key']==f'Positions_{date_filter}')]

    fig = go.Figure(go.Indicator(
    mode = "number+delta",
    value = test_df['Current Value'].sum(),
    title = {"text": "Current Portfolio Value<br>",'font':{'size':18,'color':'white'}},
    number = {'prefix': "$",'font':{'size':18,'color':'white'}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': test_df['Basis'].sum(), 'relative': True,'font':{'size':14},'position' : "right"}
        ))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    return fig

# Creating callback for twitter sentiment
@app.callback(dash.dependencies.Output('portfolio-table','children'),
    [dash.dependencies.Input('data_filter','value'),
     dash.dependencies.Input('model_filter','value')])
# Step 3: Define the graph with plotly express
def portfolio_table(value,model):

    # For testing portfolio
    mod_filter = model_dict[model]

    open_pos_df_chart = open_pos_df[open_pos_df['model']==mod_filter]
    # Getting max date for input, may make configurable later
    date = open_pos_df_chart['key'].values[-1][-10:]

    # date_filter = '2020-09-04'
    date_filter = date

    df = open_pos_df[(open_pos_df['model']==mod_filter)&\
                      (open_pos_df['key']==f'Positions_{date_filter}')][['Ticker','Current Value','% Gain']]

    table = DataTable(
        id='portfolio_table',
        data=df.to_dict('records'),
        columns = [
            dict(id='Ticker', name='Ticker'),
            dict(id='Current Value', name='Current Value', type='numeric', format=FormatTemplate.money(2)),
            dict(id='% Gain', name='% Gain', type='numeric', format=FormatTemplate.percentage(0))
            ],
        style_cell=dict(textAlign='center',fontSize=12),
        style_header=dict(backgroundColor="#191970",color='white'),
        style_data=dict(backgroundColor="gray",color='black'),

        # Setting conditional styles for sentiment and trends
         style_data_conditional=[
            {'if': {'filter_query': '{% Gain} > 0','column_id': '% Gain'},
                'color': '#03CD1E'},
            {'if': {'filter_query': '{% Gain} <= 0','column_id': '% Gain'},
                'color': '#FF0000'}],
        page_action='none',
        style_table={'height': '225px', 'overflowY': 'auto'},

        style_as_list_view=True
    )

    return table
