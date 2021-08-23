
import dash_html_components as html

desc_dict = {'Random Forest Regressor 120/30':
                [html.H4('Random Forest Regressor 120/30',style={'color':'white','fontWeight':'bold'}),
                    html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                    html.P("""
                        This model uses a GridSearch optimized Random Forest Regressor to predict
                        stock prices for the top 5 traded (by volume) stocks in each sector 120 days in the future.
                        The model re-trains itself every 30 days after receiving new data,
                        and buys/sells on the next available trading day. It decides to buy or sell dependent on % change in price,
                        and stocks are only bought and sold if they breach our thresholds.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                    html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                    html.P("""
                        Currently, the model uses a collection of Technical Trading Indicators as features, which are commonly
                        used by day traders to predict price movement. Approximately 100 features were considered for the model,
                        but only the top features that had a minimum importance of at least 2% were considered. This reduced total features
                         down to 20. Given the nature of Technical Trading Indicators, which are calculated on price movement and volume alone,
                         we still had fairly high correlation for reduced features.
                        """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                    ],
                'Random Forest Regressor 120/60':
                                [html.H4('Random Forest Regressor 120/60',style={'color':'white','fontWeight':'bold'}),
                                    html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                    html.P("""
                                        This model uses a GridSearch optimized Random Forest Regressor to predict
                                        stock prices for the top 5 traded (by volume) stocks in each sector 120 days in the future.
                                        The model re-trains itself every 60 days after receiving new data,
                                        and buys/sells on the next available trading day. It decides to buy or sell dependent on % change in price,
                                        and stocks are only bought and sold if they breach our thresholds.
                                        """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                    html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                    html.P("""
                                        Currently, the model uses a collection of Technical Trading Indicators as features, which are commonly
                                        used by day traders to predict price movement. Approximately 100 features were considered for the model,
                                        but only the top features that had a minimum importance of at least 2% were considered. This reduced total features
                                         down to 20. Given the nature of Technical Trading Indicators, which are calculated on price movement and volume alone,
                                         we still had fairly high correlation for reduced features.
                                        """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                    ],
                'Random Forest Regressor 60/30':
                                    [html.H4('Random Forest Regressor 60/30',style={'color':'white','fontWeight':'bold'}),
                                        html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            This model uses a GridSearch optimized Random Forest Regressor to predict
                                            stock prices for the top 5 traded (by volume) stocks in each sector 60 days in the future.
                                            The model re-trains itself every 30 days after receiving new data,
                                            and buys/sells on the next available trading day. It decides to buy or sell dependent on % change in price,
                                            and stocks are only bought and sold if they breach our thresholds.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                        html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            Currently, the model uses a collection of Technical Trading Indicators as features, which are commonly
                                            used by day traders to predict price movement. Approximately 100 features were considered for the model,
                                            but only the top features that had a minimum importance of at least 2% were considered. This reduced total features
                                             down to 20. Given the nature of Technical Trading Indicators, which are calculated on price movement and volume alone,
                                             we still had fairly high correlation for reduced features.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                        ],
                'Random Forest Regressor 7/7':
                                    [html.H4('Random Forest Regressor 7/7',style={'color':'white','fontWeight':'bold'}),
                                        html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            This model uses a GridSearch optimized Random Forest Regressor to predict
                                            stock prices for the top 5 traded (by volume) stocks in each sector 7 days in the future.
                                            The model re-trains itself every 7 days after receiving new data,
                                            and buys/sells on the next available trading day. It decides to buy or sell dependent on % change in price,
                                            and stocks are only bought and sold if they breach our thresholds.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                        html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            Currently, the model uses a collection of Technical Trading Indicators as features, which are commonly
                                            used by day traders to predict price movement. Approximately 100 features were considered for the model,
                                            but only the top features that had a minimum importance of at least 2% were considered. This reduced total features
                                             down to 20. Given the nature of Technical Trading Indicators, which are calculated on price movement and volume alone,
                                             we still had fairly high correlation for reduced features.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                        ],
                'CNN Image Pattern Recognition':
                                    [html.H4('CNN Image Pattern Recognition',style={'color':'white','fontWeight':'bold'}),
                                        html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            This model uses a Convolutional Neural Network to recognize patterns in images to predict
                                            how the market will perform within the upcoming few minutes. The model is trained on the
                                            previous 1 year to when back-testing begins. The trading strategy in this model is based on
                                            trades every 15 minutes, and not trading with any money greater than the initial value, as
                                            to decrease losses.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                        html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            The model uses images of intraday price candlestick charts, included with moving averages and
                                            Bollinger bands to display trends for pattern detection to classify price movements for the next
                                            15 minutes.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                        ],
                'Multi Factor Multi-Layer Perceptron':
                                [html.H4('Multi Factor MLP',style={'color':'white','fontWeight':'bold'}),
                                    html.P('Description:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            The Multi-Factor Multi-Layer Perceptron model relies on features that are broadly
                                            classified into one of five categories: 1) Valuation, 2) Quality, 3) Volatility, 4) Size, 5) Momentum.
                                            The Multi factor multi-layer perceptron model uses these features as inputs into a binary classification
                                            model design to answer one simple question, "Will the future one-year total return on this stock be in
                                            the top 50% of all S&P 500 stocks or not?"
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2,'marginBottom':'5px'}),
                                        html.P('Feature Representation:',style={'color':'white','fontWeight':'bold','fontSize':14}),
                                        html.P("""
                                            The model uses 69 features in all broadly classified as follows:
                                            Within the Value category the model uses valuation ratios like P/E, P/B, EV/EBITDA, and EV/Sales as well as standard accounting metrics EPS, Diluted EPS, EPS excluding extraordinary items, and others. Within the Quality category the model uses metrics like ROE, ROA, Net operating Assets, total accruals, and other accounting metrics.
                                            Within Volatility the model uses rolling standard deviations of a stock over various time frames.
                                            Within Momentum - The model uses exponentially weighted moving averages for a stock price over multiple timeframes as well as historic returns over multiple time frames.
                                            The Size factor - is represented average market cap.
                                            """,style={'color':'white','fontSize':12,'lineHeight':1.2})
                                        ]

                }
