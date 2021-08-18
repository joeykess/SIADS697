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


layout = html.Div([
                dbc.Row([
                    dbc.Col(id='jupyter-html',width=12),
                    ],style={'width':'100%'})
                ])

@app.callback(dash.dependencies.Output('jupyter-html','children'),
    [dash.dependencies.Input('model_filter','value')])
def html_filter(model_filter):

    if model_filter[:6] == 'Random':

        return html.Iframe(src='assets/Tyler RF Regressor Model-Port_2.html',width='100%',height=1000)

    if model_filter == 'Multi Factor Multi-Layer Preceptron':

        return None

    if model_filter == 'CNN Image Pattern Recognition':

        return None
