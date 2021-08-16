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
from dash.exceptions import PreventUpdate
from app import app
from port_charts import *

# model_dict = {'Random Forest Regressor 120/30': 'RF Reg_target_120_rebal_30_2017-01-01',
#               'Random Forest Regressor 120/60': 'RF Reg_target_120_rebal_60_2017-01-01'
#               # 'Random Forest Regressor 60/30': 'RF Reg_target_60_rebal_30_2017-01-01'
#               # 'CNN Visual Pattern Recognition': '75percent_confidence_no_holding_15m_cnn'
#              }
# model_list = [key for key in model_dict.keys()]

# Loading model stat files for Tyler
r2_df = pd.read_csv('assets/models/tyler_rf_daily_update/r2_df.csv',index_col=0)
feat_df = pd.read_csv('assets/models/tyler_rf_daily_update/feature_importance.csv',index_col=0)
corr_df = pd.read_csv('assets/models/tyler_rf_daily_update/corr.csv',index_col=0)

# Creating layout for Tyler's model metrics
layout_tyler = html.Div([
                dcc.Graph(id='model-metrics',style={'width':'100%','borderBottom':'thin lightgrey solid',
                                    'height':'25%','float':'top','marginBottom':'5px'}),
                # dcc.Graph(id='corr-chart',style={'width':'100%','display':'inline-block'}),
                dcc.Graph(id='feature-chart',style={'width':'100%','marginTop':'5px'})
            ],style={'width':'100%','height':350,'overflowY':'auto'})

# Loading model stat data for Jeff
csv_path = 'assets/models/jeff_multi_factor/mf_mlp.csv'

# Creating layout for Jeff's model metrics
layout_jeff = html.Div([
                dcc.Graph(id='model-metrics-jeff',style={'width':'100%','height':'45%','marginBottom':'5px'}),
                dcc.Graph(id='model-metrics-jeff2',style={'width':'100%','height':'45%','marginBottom':'5px'})

                ],style={'width':'100%','height':'350','overflowY':'auto'})


@app.callback(dash.dependencies.Output('model-metrics','figure'),
    [dash.dependencies.Input('model_filter','value')])
def model_metrics(model_filter):

    if model_filter == 'Random Forest Regressor 60/30':
        target = 'Feats 60' # May come back and fix...
    if model_filter == 'Random Forest Regressor 7/7':
        target = 'Target 7'
    else:
        target = 'Target 120'

    fig = go.Figure(go.Indicator(
                        mode = "number",
                        value = r2_df[target].values[0],
                        title = {"text": "Model R<sup>2</sup> Score:<br>",'font':{'size':18,'color':'white'}},
                        number = {'font':{'size':18,'color':'white'},'valueformat':'%'},
                        domain = {'x': [0, 1], 'y': [0, 1]}
                        ))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')

    return fig

@app.callback(dash.dependencies.Output('feature-chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def model_metrics(model_filter):

    if model_filter == 'Random Forest Regressor 60/30':
        target = 'Feats 60' # May come back and fix...
    if model_filter == 'Random Forest Regressor 7/7':
        target = 'Target 7'
    else:
        target = 'Target 120'

    feats_df = feat_df[[target,'Features']].sort_values(by=target,ascending=False)

    fig = go.Figure()

    fig = fig.add_trace(
            go.Bar(y=feats_df[target].values, x=feats_df['Features'].values)
            )
    fig.update_layout(
            title= dict(text=f'Feature Importance',font = dict(size = 18, color = 'white'),x = 0.5, y = .99),
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(color='white',size=10),
            height = 200)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    return fig

@app.callback(dash.dependencies.Output('corr-chart','figure'),
    [dash.dependencies.Input('model_filter','value')])
def model_metrics(model_filter):

    fig = px.imshow(corr_df)

    fig.update_layout(title='Feature Correlation')

    return fig

# Creating figure for Jeff's model
@app.callback(dash.dependencies.Output('model-metrics-jeff','figure'),
    [dash.dependencies.Input('model_filter','value')])
def model_metrics_jeff(model_filter):

    fig = mlp_stat_chart(csv_path, stat = 'loss')

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#A9A9A9')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False,range=[0, 155])

    return fig

# Creating figure for Jeff's model - 2
@app.callback(dash.dependencies.Output('model-metrics-jeff2','figure'),
    [dash.dependencies.Input('model_filter','value')])
def model_metrics_jeff(model_filter):

    fig = mlp_stat_chart(csv_path, stat = 'binary_accuracy')

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#A9A9A9')
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False,range=[0, 155])

    return fig
