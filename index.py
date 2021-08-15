# Dash modules
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Load initial app.py
from app import app
from app import server

# Import pages creates separately
from apps import portfolio_performance
from apps import model_performance
from apps.ind_css import *

title_style = {'display': 'inline-block',
               'textAlign':'left',
               'verticalAlign':'center',
               'lineHeight':2.5,
               'height':75,
               'color':'white',
               'fontSize':32,
               'margin-left':5,
               'font-weight':'bold',
               # 'border': 'thick black solid',\
               'width':'85%'}

title_link_style = {'display': 'inline-block',
               'textAlign':'center',
               'vertical-align':'center',
               'lineHeight':2.6,
               'height':75,
               'width':'10%',
               'float':'right',
               'margin-top':5,
               # 'border': 'thick black solid',\
               'fontSize':12,}
               # 'backgroundColor': 'rgb(212, 150, 18)'}

tabs_style = {
             'textAlign':'center',
             'verticalAlign':'top',
             'width':'100%',
             'height':50,
             'backgroundColor':'gray',
             'display': 'inline-block',
             # 'border': 'thin lightgrey solid',
             'fontSize':15}
             # 'height':15}

blank_tab_style = {'display': 'inline-block',
                   'width':'15%',
                   'textAlign':'center',
                   'verticalAlign':'top',
                   'float':'right',
                   'border': 'thin lightgrey solid',
                   'margin': 0,
                   'fontSize':10,
                   'height':20}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#4e5d6c',
    'color':'white',
    'lineHeight':2.5,
    'fontSize':15
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#191970',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'lineHeight':2.5,
    'fontSize':15
}

model_dict = {'Random Forest Regressor 120/30': 'RF Reg_target_120_rebal_30_2017-01-01',
              'Random Forest Regressor 120/60': 'RF Reg_target_120_rebal_60_2017-01-01',
              'Random Forest Regressor 60/30': 'RF Reg_target_60_rebal_30_2017-01-01',
              'Random Forest Regressor 7/7': 'RF Reg_target_7_rebal_7_2017-01-01',
              'Multi Factor Multi-Layer Preceptron': 'MF_MLP'
              # 'CNN Visual Pattern Recognition': '75percent_confidence_no_holding_15m_cnn'
             }
model_list = [key for key in model_dict.keys()]

app.layout = html.Div([dcc.Store(id='memory-output',storage_type='local'),

    html.Div([
        html.H1('Financial Modeling Exploration Dashboard',style=title_style),
        html.Div([dbc.Button('Dashboard Info Link',id="open", n_clicks=0),
                  dbc.Modal([
                    dbc.ModalHeader("Legal Disclaimer",style={"color":'black'}),
                    dbc.ModalBody("""The content of this site is for informational purposes only.
                    There is risk in trading in securities of any kind, and we will not be held responsible for any losses that occur.
                    """,style={'color':'white'}),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ml-auto", n_clicks=0))
                  ],id="modal",centered=True,is_open=False)
        ],style=title_link_style)
        ],style={'height':75,'margin-bottom':10}),

    html.Div([
        html.Div([
            dcc.Tabs(id='tabs-example', value='tab-1', children=[
                dcc.Tab(label='Portfolio Performance', value='tab-1',style=tab_style,selected_style=tab_selected_style),
                dcc.Tab(label='Model Performance', value='tab-2',style=tab_style,selected_style=tab_selected_style),
                dcc.Tab(label='Tab three', value='tab-3',style=tab_style,selected_style=tab_selected_style)
                ],style=tabs_style)],style={'display':'inline-block','width':'45%'}),
        html.Div([
            html.A('Pick a Model:',style={'color':'white','display':'inline-block','width':'25%','verticalAlign':'middle','textAlign':'right','marginRight':'10px'}),
            dcc.Dropdown(id='model_filter',
                options=[{'label': i, 'value': i} for i in model_list],
                value='Random Forest Regressor 120/30',clearable=False,style={'display': 'inline-block','width':'70%','verticalAlign':'top'}
                )],style={'display':'inline-block','width':'30%','height':'100%','verticalAlign':'top','float':'middle'}),
        html.Div([
            html.A('Pick a Date:',style={'color':'white','display':'inline-block','width':'25%','verticalAlign':'middle','textAlign':'right','marginRight':'10px'}),
            dcc.Dropdown(id='date_filter',
                options=[{'label': i, 'value': i} for i in ['2021-03-03','2021-03-01']],value='2021-03-03',
                        style={'display': 'inline-block','verticalAlign':'top','width':'70%'})
                ],style={'display':'inline-block','width':'25%','height':'100%','verticalAlign':'top','float':'right'})
        ],style={'width':'100%'}),

    html.Div(id='tabs-example-content',style={'border': 'thin lightgrey solid'})
])

@app.callback(dash.dependencies.Output('tabs-example-content', 'children'),
              [dash.dependencies.Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return portfolio_performance.layout

    elif tab == 'tab-2':
        return model_performance.layout

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab 3 content')
        ])

# Callback to open disclaimer modal
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback to add filter details to memory
@app.callback(dash.dependencies.Output('memory-output','data'),
              [dash.dependencies.Input('model_filter', 'value')])
def filter_model(value):
    return {'model_to_filter': value}

if __name__ == '__main__':
    app.run_server(debug=True)
