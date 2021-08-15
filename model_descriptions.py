
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
                'CNN Visual Pattern Recognition':['TBD'],
                'Random Forest Regressor 7/7':['TBD'],
                'Multi Factor Multi-Layer Preceptron':
                                [html.H4('Multi Factor Multi-Layer Preceptron',style={'color':'white'}),
                                    html.P('Description:',style={'color':'white','fontWeight':'bold'}),
                                        html.P("""
                                            The Multi-Factor Multi-Layer Perceptron model relies on features that are broadly
                                            classified into one of five categories: 1) Valuation, 2) Quality, 3) Volatility, 4) Size, 5) Momentum.
                                            The Multi factor multi-layer perceptron model uses these features as inputs into a binary classification
                                            model design to answer one simple question, "Will the future one-year total return on this stock be in
                                            the top 50% of all S&P 500 stocks or not?"
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                        html.P('Feature Representation:',style={'color':'white','fontWeight':'bold'}),
                                        html.P("""
                                            The model uses 69 features in all broadly classified as follows:
                                            Within the Value category the model uses valuation ratios like P/E, P/B, EV/EBITDA, and EV/Sales as well as standard accounting metrics EPS, Diluted EPS, EPS excluding extraordinary items, and others. Within the Quality category the model uses metrics like ROE, ROA, Net operating Assets, total accruals, and other accounting metrics.
                                            Within Volatility the model uses rolling standard deviations of a stock over various time frames.
                                            Within Momentum - The model uses exponentially weighted moving averages for a stock price over multiple timeframes as well as historic returns over multiple time frames.
                                            The Size factor - is represented average market cap.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                        ]

                }
