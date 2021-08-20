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
from yahoo_fin import stock_info as si
import financial_metrics as fm
from port_charts import *
from model_descriptions import *
from apps import model_stats

from sqlalchemy import create_engine
import psycopg2
from psycopg2 import connect

def import_track_record():
    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = "SELECT * FROM track_record"
    data = pd.read_sql_query(query,conn)
    data = data.sort_values(['Date', 'model'])
    return data
track_df = import_track_record().drop('index',axis=1)

def import_open_positions():
    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = "SELECT * FROM open_positions"
    data = pd.read_sql_query(query,conn)
    data = data.sort_values(['key', 'model'])
    return data
open_pos_df = import_open_positions().drop('index',axis=1)

def import_cash_record():
    conn = connect(dbname = '697_temp', user = 'postgres', host = 'databasesec.cvhiyxfodl3e.us-east-2.rds.amazonaws.com', password = 'poRter!5067')
    cur = conn.cursor()
    query = "SELECT * FROM cash_record"
    data = pd.read_sql_query(query,conn)
    data = data.sort_values(['model','key'])
    return data
cash_df = import_cash_record().drop('index',axis=1)

model_dict = {'Random Forest Regressor 120/30': 'RF Reg_target_120_rebal_30_2018-01-01',
              'Random Forest Regressor 120/60': 'RF Reg_target_120_rebal_60_2018-01-01',
              'Random Forest Regressor 60/30': 'RF Reg_target_60_rebal_30_2018-01-01',
              'Random Forest Regressor 7/7': 'RF Reg_target_7_rebal_7_2018-01-01',
              'Multi Factor Multi-Layer Perceptron': 'MF_MLP',
              'CNN Image Pattern Recognition': '75percent_confidence_no_holding_15m_cnn'
             }
model_list = [key for key in model_dict.keys()]

layout = html.Div([
            dbc.Row(
                [
                dbc.Col(html.Div(id='model-stats'),width=3,style={'margin':'5px','height':300}),
                dbc.Col(dcc.Graph(id='perf_chart',style=chart_style_dbc),style={'width':'45%','vertical-align':'top','float':'middle','margin':'5px','height':300}),
                dbc.Col(html.Div(id='mod_desc',style={'padding':'5px','width':'100%'}),width=3,style={'border':'thin lightgrey solid','borderRadius':'.4rem','margin':'5px','overflowY':'auto','height':300}),
                ],style={'width':'100%'}),
            dbc.Row(
                [
                dbc.Col(dcc.Graph(id='sector_chart',style=chart_style_dbc),width=3,style={'margin':'5px','height':300}),
                dbc.Col(dcc.Graph(id='risk_adj_chart',style=chart_style_dbc),style={'width':'45%','float':'middle','margin':'5px','height':300}),
                dbc.Col(dcc.Graph(id='risk_return_chart',style=chart_style_dbc),width=3,style={'width':'25%','float':'right','margin':'5px','height':300}),
                ],style={'width':'100%'})
            ],style={'width':'100%'})

@app.callback(dash.dependencies.Output('store_callback','children'),
             [dash.dependencies.Input('memory-output','data')],
             [dash.dependencies.State('memory-output','data')])
def test_store(data,data2):
    return data2['model_to_filter']

@app.callback(dash.dependencies.Output('perf_chart','figure'),
             [dash.dependencies.Input('model_filter','value')])
def perf_chart_func(model_filter):

    # Getting model from session (or index page)
    mod_filter = model_dict[model_filter]

    chart_df = track_df[track_df['model']==mod_filter]

    fig = performance_chart(chart_df, 'spy')

    fig.update_layout(template="ggplot2",
                      hovermode="x unified",
                      showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='#A9A9A9',
                      xaxis_showgrid=False,
                      font=dict(size=10,color='white'))

    return fig

@app.callback(dash.dependencies.Output('sector_chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def sector_chart_func(model_filter):

    # Getting model from session (or index page)
    # model_filter = session_data['model_to_filter']
    mod_filter = model_dict[model_filter]

    open_pos_df_chart = open_pos_df[open_pos_df['model']==mod_filter]
    cash_df_chart = cash_df[cash_df['model']==mod_filter]

    # Getting max date for input, may make configurable later
    date = open_pos_df_chart['key'].values[-1][-10:]
    pos_key = f'Positions_{date}'
    cash_key = f'cash_{date}'

    fig = sector_plot(open_pos_df_chart[open_pos_df_chart['key']==pos_key],\
            cash_df_chart[cash_df_chart['key']==cash_key].cash.values[0],date)

    fig.update_layout(template="ggplot2",
                      paper_bgcolor='rgba(0,0,0,0)'
                      )
    return fig

@app.callback(dash.dependencies.Output('risk_adj_chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def risk_adj_func(model_filter):

    mod_filter = model_dict[model_filter]
    chart_df = track_df[track_df['model']==mod_filter]

    fig = risk_adjusted_metrics(chart_df,'spy')

    fig.update_layout(template="ggplot2",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='#A9A9A9',
                      font=dict(size=10,color='white'),
                      xaxis_showgrid=False)

    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))

    return fig

@app.callback(dash.dependencies.Output('risk_return_chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def risk_return_func(model_filter):

    mod_filter = model_dict[model_filter]
    chart_df = track_df[track_df['model']==mod_filter]

    fig = risk_to_ret(chart_df,'spy')

    fig.update_layout(template="ggplot2",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='#A9A9A9',
                      font=dict(size=10,color='white'),
                      xaxis_showgrid=False,
                      margin=dict(l=10, r=10, t=50, b=10))

    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
    fig.update_yaxes(tickangle=-45)

    return fig

@app.callback(dash.dependencies.Output('mod_desc','children'),
    [dash.dependencies.Input('model_filter','value')])
def model_desc(model_filter):

    layout2 = desc_dict[model_filter]

    return layout2

@app.callback(dash.dependencies.Output('model-stats','children'),
    [dash.dependencies.Input('model_filter','value')])
def model_stat(model_filter):

    if model_filter[:6] == 'Random':

        return model_stats.layout_tyler

    if model_filter == 'Multi Factor Multi-Layer Perceptron':

        return model_stats.layout_jeff

    if model_filter == 'CNN Image Pattern Recognition':

        return model_stats.layout_joey
