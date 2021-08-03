<h1>Convolutional Neural Network (CNN) Trading Model </h1>

<h3>Intent</h3>
<p>
This modeling process was meant to evaluate [candlestick charts](https://www.investopedia.com/trading/candlestick-charting-what-is-it/) and utilize CNNs to read and interpret the chart to know if the price will go up or down in the following <i>x</i> minutes.
</p>

#### Hypothesis:
<p>
Day traders typically monitor the markets every minute that the exchanges are open. 


Given traders frequently make decisions on purchasing assets based on technical indicators, or statistics about the movement of stock prices and trading activity, I hopthesize that these decisions can be modeled using advanced modeling techniques and hopefully identified before actual price movement happens. Automating this via machine learning should allow us to mechanically trade faster than a normal human using manual techniques.
</p>

#### Data Used:
1. Historical intraday stock prices at 5min intervals from the past 2 years for stocks not on NASDAQ.
   1. 10 stocks and 2 ETFs were chosen with high volume: NVDA, AMD, JPM, JNJ, MRNA, F, TSLA, MSFT, BAC, BABA, SPY, QQQ
2. Data for training was 1st year, and back tested data is previous year to now.

#### Modeling Process:
1. `short_chart_creator.py` create candlesticks using the `mplfinance` Python library
2. These charts then split into buy or hold (the "not buying right now" signal) and then re-split into training, validation, testing
3. `short_cnn.py` trains the CNN model on the candlesticks
   1. If the stock goes up, then "buy" class, if not then "hold" class
4. `intraday_portfolio.py` has a portfolio class to keep track of the back testing process, and has the model predict the classes of charts created throughout the year of back testing

#### Other Considerations / Next Steps:
1. Adding other classification models for shorting and holding stocks instead of purchasing and selling in <i>x</i> minute intervals
2. Buying and selling neglected price to trade on specific platforms which is why holding stocks is not done, so in the future this should be included
3. Extending this modeling to a higher frequency for 1 minute trading