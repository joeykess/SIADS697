## Random Forest Regression Trading Model

#### Intent:
<p>
This modeling process was meant to test techniques that approximate a target price of multiple stocks in the S&P 500 at some point in the future. The intent is to include technical indicators, which are used by traders to uncover trends into future stock performance.
</p>

#### Hypothesis:
<p>
Given traders frequently make decisions on purchasing assets based on technical indicators, or statistics about the movement of stock prices and trading activity, I hopthesize that these decisions can be modeled using advanced modeling techniques and hopefully identified before actual price movement happens. Automating this via machine learning should allow us to mechanically trade faster than a normal human using manual techniques.
</p>

#### Data Used:
1. Historical stock prices (from 1980 forward) for every stock in the S&P 500
2. Data was truncated to 2014 forward to account for space and performance limitations
3. Data features were created using the Technical Analysis python library, but we also created our own functions to create technical features

#### Modeling Process:
1. As stated above, features were created solely from the Technical Analysis python library
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
1. Sector weighting should be included to ensure portfolio is appropriately risk-weighted
2. The buy/sell strategy was basic in practice, as it only considered individual point-in-time estimate, which can lead to <b>following bad trends? something about not selling at consistent <5% drops</b>. A more advanced strategy would take into a look back strategy of prior prices at re-balance intervals
3. Fundamental features should be added to assess the actual performance of companies we are trading in, as this is a better indication of future performance
