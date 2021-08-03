<h1>Convolutional Neural Network (CNN) Trading Model </h1>

<h3>Intent</h3>
<p>
This modeling process was meant to evaluate [candlestick charts](https://www.investopedia.com/trading/candlestick-charting-what-is-it/) and utilize CNNs to read and interpret the chart to know if the price will go up or down in the following <i>x</i> minutes.
</p>

#### Hypothesis:
<p>
Day traders  

Given traders frequently make decisions on purchasing assets based on technical indicators, or statistics about the movement of stock prices and trading activity, I hopthesize that these decisions can be modeled using advanced modeling techniques and hopefully identified before actual price movement happens. Automating this via machine learning should allow us to mechanically trade faster than a normal human using manual techniques.
</p>

#### Data Used:
1. Historical intraday stock prices at 5min intervals from the past 2 years for stocks not on NASDAQ.
   1. 10 stocks and 2 ETFs were chosen with high volume: NVDA, AMD, JPM, JNJ, MRNA, F, TSLA, MSFT, BAC, BABA, SPY, QQQ
2. Data for training was 1st year, and back tested data is previous year to now.

#### Modeling Process:
1. Candlesticks were created in the `short_chart_creator.py` file using the `mplfinance` Python library
2. Some basic data visualizations were created using Plotly to ensure data was uncorrupted and reasonable
3. A basic and unoptimized Random Forest Regressor model was created to score features on importance
4. Features were selected based on importance and only features adding at least 1% importance were selected. This accounted for 15? of XX features
5. Given the nature of Technical Analysis features, high correlation is expected, but correlation was checked anyways - <b>Currently not dealt with</b>
6. A grid search was also performed to uncover an optimal configuration for our model
7. To get a sense of strong target variables, or future price prediction, some testing was done on multiple targets to see how effective prediction was. Target variables used were 7, 30, 60, 120 Day future price predictions
8. All targets had great accuracy for initial predictions, but performance degraded the further in the future predictions got from training data
9. Given the degredation over time, a daily re-training process was used to continually use new information to inform the next prediction. This proved to be effective, as all models achieved >96% R^2 scores on training fit accuracy. This process was used to predict prices and buy/sell stock
10. Maybe the most difficult part was actually creating the appropriate buying and selling strategy. Our team created a portofolio class that was used to manage a portfolio, and buying and selling of stock. Thresholds were set to "re-balance" our portfolio on regular intervals given a certain predicted % change in price (5% for all testing)
11. The algorithm created looped through re-balance intervals (manually configured), re-fit the model on new data, then sold and bought new stock according to results
12. Stock was bought and sold in order of predicted % change. If funds weren't sufficient, the algorithm bought only how much could be afforded

#### Other Considerations / Next Steps:
1. Adding other classification models for shorting and holding stocks instead of purchasing and selling in <i>x</i> minute intervals
2. Buying and selling neglected price to trade on specific platforms which is why holding stocks is not done, so in the future this should be included
3. 