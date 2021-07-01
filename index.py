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
from apps.ind_css import *

title_style = {'display': 'inline-block',
               'textAlign':'left',
               'verticalAlign':'center',
               'lineHeight':2.5,
               'height':75,
               'border': 'thick black solid',\
               'width':'85%',
               'backgroundColor': 'rgb(212, 150, 18)'}

title_link_style = {'display': 'inline-block',
               'textAlign':'center',
               'vertical-align':'center',
               'lineHeight':2.6,
               'height':75,
               'width':'10%',
               'float':'right',
               # 'border': 'thick black solid',\
               'fontSize':12,}
               # 'backgroundColor': 'rgb(212, 150, 18)'}

tabs_style = {
             'textAlign':'center',
             'verticalAlign':'top',
             'width':'50%',
             'height':50,
             'backgroundColor':'gray',
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
    'lineHeight':2
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'gray',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold',
    'lineHeight':2
}

app.layout = html.Div([

    html.Div([
        html.H1('Dashboard Title',style=title_style),
        html.Div([dbc.Button('Dashboard Info Link',id="open", n_clicks=0),
                  dbc.Modal([
                    dbc.ModalHeader("Header",style={"color":'black'}),
                    dbc.ModalBody("This is the content of the modal"),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ml-auto", n_clicks=0))
                  ],id="modal",centered=True,is_open=False)
        ],style=title_link_style)
        ]),

    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Portfolio Performance', value='tab-1',style=tab_style,selected_style=tab_selected_style),
        dcc.Tab(label='Model Performance', value='tab-2',style=tab_style,selected_style=tab_selected_style),
        dcc.Tab(label='Tab three', value='tab-3',style=tab_style,selected_style=tab_selected_style)
    ],style=tabs_style),
    # html.H2('Hello!',style={'border': 'thin lightgrey solid'})
    html.Div(id='tabs-example-content',style={'border': 'thin lightgrey solid'})
])

@app.callback(dash.dependencies.Output('tabs-example-content', 'children'),
              [dash.dependencies.Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return portfolio_performance.layout

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab 2 content ')
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab 3 content')
        ])


# modal = html.Div(
#     [
#         dbc.Button("Open modal", id="open", n_clicks=0),
#         dbc.Modal(
#             [
#                 dbc.ModalHeader("Header"),
#                 dbc.ModalBody("This is the content of the modal"),
#                 dbc.ModalFooter(
#                     dbc.Button(
#                         "Close", id="close", className="ml-auto", n_clicks=0
#                     )
#                 ),
#             ],
#             id="modal",
#             is_open=False,
#         ),
#     ]
# )
#
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
