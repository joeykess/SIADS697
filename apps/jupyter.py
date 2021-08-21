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
                    dbc.Col(id='notebook-title',width=12),
                        ]),

                dbc.Row([
                    dbc.Col(id='jupyter-html',width=12),
                        ])
            ],style={'width':'100%','float':'middle','border':'thin black solid'})


# Creating filter to load different Jupyter Notebook Titles
@app.callback(dash.dependencies.Output('notebook-title','children'),
    [dash.dependencies.Input('model_filter','value')])
def html_filter(model_filter):

    if model_filter[:6] == 'Random':

        return html.H1('Random Forest Regressor',
                            style={'color':'white','textAlign':'center'})

    if model_filter == 'Multi Factor Multi-Layer Perceptron':

        return html.H1('Multi Factor Multi-Layer Perceptron',
                            style={'color':'white','textAlign':'center'})

    if model_filter == 'CNN Image Pattern Recognition':

        return html.H1('CNN Image Pattern Recognition',
                            style={'color':'white','textAlign':'center'})

# Creating filter to load different Jupyter Notebooks
@app.callback(dash.dependencies.Output('jupyter-html','children'),
    [dash.dependencies.Input('model_filter','value')])
def html_filter(model_filter):

    if model_filter[:6] == 'Random':

        return html.Div([html.Iframe(src='assets/Tyler RF Regressor Model-Port_2.html',
                                style={'width':'100%','height':1000,'display':'flex'})
                            ])

    if model_filter == 'Multi Factor Multi-Layer Perceptron':

        return html.Div([html.Iframe(src='assets/mf_mlp.html',
                                style={'width':'100%','height':1000,'display':'flex'})
                            ])
        # return None # Returning None for now, as Jeff's file is too large

    if model_filter == 'CNN Image Pattern Recognition':

        return html.Div([html.Iframe(src='assets/short_cnn_notebook.html',
                                style={'width':'100%','height':1000,'display':'flex'})
                            ])
