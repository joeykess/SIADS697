from app import app
from app import server
import dash

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

    return html.Div([html.H2(f'News for {ticker}',style={'backgroundColor':'gray'}),\
                dbc.ListGroup([
                    dbc.ListGroupItem([html.Div([
                            html.A(html.P(news[0]['title'],style=news_style),\
                            href=(news[0]['link'])),\
                            html.A(html.P(news[0]['published'],style=news_style))
                            ])
                      ],color='gray'),\
                dbc.ListGroupItem([html.Div([
                            html.A(html.P(news[1]['title'],style=news_style),\
                            href=(news[1]['link'])),\
                            html.A(html.P(news[1]['published'],style=news_style))
                            ])
                      ],color='gray'),\
                dbc.ListGroupItem([html.Div([
                            html.A(html.P(news[2]['title'],style=news_style),\
                            href=(news[2]['link'])),\
                            html.A(html.P(news[2]['published'],style=news_style))
                            ])
                      ],color='gray'),\
                dbc.ListGroupItem([html.Div([
                            html.A(html.P(news[3]['title'],style=news_style),\
                            href=(news[3]['link'])),\
                            html.A(html.P(news[3]['published'],style=news_style))
                            ])
                      ],color='gray'),\
                dbc.ListGroupItem([html.Div([
                            html.A(html.P(news[4]['title'],style=news_style),\
                            href=(news[4]['link'])),\
                            html.A(html.P(news[4]['published'],style=news_style))
                            ])
                      ],color='gray'),\
                ],flush=True)
            ])
