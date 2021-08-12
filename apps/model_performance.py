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
import pickle

port_2 = pickle.load(open( "RF_Reg_target_120_rebal_30_2017-01-01.pkl", "rb" ))

layout = html.Div([

            dbc.Row(
                [
                dbc.Col(html.A('What Models Do You Want to Compare?',style={'margin':'5px','lineHeight':2}),width=2),
                dbc.Col(dcc.Dropdown(id='model_filter',
                    options=[{'label': i, 'value': i} for i in ['Random Forest Regressor','Next Model']],
                    value='Random Forest Regressor'),width=3,style={'margin':'5px','lineHeight':2}),
                dbc.Col(html.A('Pick Date to Analyze',style={'margin':'5px','lineHeight':2}),width=1.5),
                dbc.Col(dcc.Dropdown(id='date_filter',
                    options=[{'label': i, 'value': i} for i in ['2021-03-03','2021-03-01']],
                    value='2021-03-03'),width=2,style={'margin':'5px','lineHeight':2})
                ]),
            dbc.Row(
                [
                dbc.Col(html.Div("R^2,ROI, etc"),width=3,style={'border': 'thin black solid','margin':'5px'}),
                dbc.Col(dcc.Graph(id='perf_chart',style=chart_style_dbc),style={'border': 'thin black solid','width':'45%','float':'middle','margin':'5px'}),
                dbc.Col(html.Div(id='mod_desc'),width=3,style={'border': 'thin black solid','width':'25%','float':'right','margin':'5px'}),
                ]),
            dbc.Row(
                [
                dbc.Col(dcc.Graph(id='sector_chart',style=chart_style_dbc),width=3,style={'border': 'thin black solid','margin':'5px'}),
                dbc.Col(dcc.Graph(id='risk_adj_chart',style=chart_style_dbc),style={'border': 'thin black solid','width':'45%','float':'middle','margin':'5px'}),
                dbc.Col(dcc.Graph(id='risk_return_chart',style=chart_style_dbc),width=3,style={'border': 'thin black solid','width':'25%','float':'right','margin':'5px'}),
                ])
            ])

@app.callback(dash.dependencies.Output('perf_chart','figure'),
    [dash.dependencies.Input('model_filter','value')])

def perf_chart_func(model_filter):

    fig = performance_chart(port_2, 'spy')

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

    fig = sector_plot(port_2, '2021-03-03')

    fig.update_layout(template="ggplot2",
                      paper_bgcolor='rgba(0,0,0,0)')

    return fig

@app.callback(dash.dependencies.Output('risk_adj_chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def risk_adj_func(model_filter):

    fig = risk_adjusted_metrics(port_2,'spy')

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

    fig = risk_to_ret(port_2,'spy')

    fig.update_layout(template="ggplot2",
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='#A9A9A9',
                      font=dict(size=10,color='white'),
                      xaxis_showgrid=False,
                      margin=dict(l=10, r=10, t=50, b=10))

    fig.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))

    return fig

@app.callback(dash.dependencies.Output('mod_desc','children'),
    [dash.dependencies.Input('model_filter','value')])
def model_desc(model_filter):

    layout2 = [html.H1('Random Forest Regressor Model',style={'color':'white'}),
                html.P('Description:',style={'color':'white','fontWeight':'bold'}),
                html.P("""
                        This model uses a GridSearch optimized Random Forest Regressor to predict
                        stock prices for the top 5 traded stocks in each sector 120 days in the future.
                        The model re-trains itself daily after recieving new data about trades from that day,
                        and buys/sells the next available trading day.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                html.P('Feature Representation:',style={'color':'white','fontWeight':'bold'}),
                html.P("""
                        Currently, the model uses a collection of Technical Trading Indicators, that are commonly
                        used by day traders to predict price movement.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                        ]
    return layout2
