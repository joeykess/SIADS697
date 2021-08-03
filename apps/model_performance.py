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

layout = html.Div([

            dbc.Row(
                [
                dbc.Col(html.A('What Models Do You Want to Compare?',style={'margin':'5px','lineHeight':2}),width=3),
                dbc.Col(dcc.Dropdown(id='model_filter',
                    options=[{'label': i, 'value': i} for i in ['Random Forest Regressor','Next Model']],
                    value='Random Forest Regressor',multi=True),width=5,style={'margin':'5px','lineHeight':2})
                ]),
            dbc.Row(
                [
                dbc.Col(html.Div("R^2,ROI, etc"),width=3,style={'border': 'thin black solid','margin':'5px'}),
                dbc.Col(html.Div("Model Performance Line Chart"),style={'border': 'thin black solid','width':'45%','float':'middle','margin':'5px'}),
                dbc.Col(html.Div("Model Description"),width=3,style={'border': 'thin black solid','width':'25%','float':'right','margin':'5px'}),
                ]),
            dbc.Row(
                [
                dbc.Col(html.Div("Stock Mix"),width=3,style={'border': 'thin black solid','margin':'5px'}),
                dbc.Col(html.Div("% Mix by Sector"),style={'border': 'thin black solid','width':'45%','float':'middle','margin':'5px'}),
                dbc.Col(html.Div("Confusion Matrix?"),width=3,style={'border': 'thin black solid','width':'25%','float':'right','margin':'5px'}),
                ])
            ])
