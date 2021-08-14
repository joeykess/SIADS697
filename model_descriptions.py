
import dash_html_components as html

desc_dict = {'Random Forest Regressor 120/30':
                [html.H4('Random Forest Regressor 120/30',style={'color':'white'}),
                    html.P('Description:',style={'color':'white','fontWeight':'bold'}),
                    html.P("""
                        This model uses a GridSearch optimized Random Forest Regressor to predict
                        stock prices for the top 5 traded stocks in each sector 120 days in the future.
                        The model re-trains itself every 30 days after receiving new data,
                        and buys/sells the next available trading day.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                    html.P('Feature Representation:',style={'color':'white','fontWeight':'bold'}),
                    html.P("""
                        Currently, the model uses a collection of Technical Trading Indicators, that are commonly
                        used by day traders to predict price movement.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                    ],
                'Random Forest Regressor 120/60':
                                [html.H4('Random Forest Regressor 120/60',style={'color':'white'}),
                                    html.P('Description:',style={'color':'white','fontWeight':'bold'}),
                                    html.P("""
                                        This model uses a GridSearch optimized Random Forest Regressor to predict
                                        stock prices for the top 5 traded stocks in each sector 120 days in the future.
                                        The model re-trains itself every 60 days after receiving new data,
                                        and buys/sells the next available trading day.
                                        """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                    html.P('Feature Representation:',style={'color':'white','fontWeight':'bold'}),
                                    html.P("""
                                        Currently, the model uses a collection of Technical Trading Indicators, that are commonly
                                        used by day traders to predict price movement.
                                        """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                    ],
                'Random Forest Regressor 60/30':['TBD'],
                'CNN Visual Pattern Recognition':['TBD']
                }
