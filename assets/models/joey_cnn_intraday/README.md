<h1>Convolutional Neural Network (CNN) Trading Model </h1>

<h3>Intent</h3>
<p>
This modeling process was meant to evaluate [candlestick charts](https://www.investopedia.com/trading/candlestick-charting-what-is-it/) and utilize CNNs to read and interpret the chart to know if the price will go up or down in the following <i>x</i> minutes.
</p>

### Hardware Specs Necessary:
All of the code ran for the CNN training and back-testing was done on the following machine:
- AMD 5950x 16 core processor
- 64GB RAM
- Nvidia RTX 3090 24GB VRAM

For any of this code running on lower specs it is entirely possible to run out of GPU or GPU memory and for the training to fail. I did not run into this instance, however the batch size can be decreased if this is the case.

#### Hypothesis:
<p>
Day traders typically monitor the markets every minute that the exchanges are open. To decrease risk, trading outside of opening hour is done in this model. I believe that we can model and predict the near future market price using a model to interpret candlesticks faster than humans can to achieve (hopefully) high returns trading.
</p>

#### Data Used:
1. Historical intraday stock prices at 5min intervals from the past 2 years for stocks not on NASDAQ.
   1. **_10 stocks_** and **_2 ETFs_** were chosen with high volume: _NVDA, AMD, JPM, JNJ, MRNA, F, TSLA, MSFT, BAC, BABA_; **_ETFS:_** _SPY, QQQ_
2. Data for training was 1st year, and back tested data is previous year to now.

#### Modeling Process:
1. `short_chart_creator.py` create candlesticks using the `mplfinance` Python library
2. These charts then split into buy or hold (the "not buying" signal) and then re-split into training, validation, testing
3. `short_cnn.py` trains the CNN model on the candlesticks
   1. If the stock goes up, then "buy" class, if not then "hold" class
4. `intraday_portfolio.py` has a portfolio class to keep track of the back testing process, and has the model predict the classes of charts created throughout the year of back testing

#### Other Considerations / Next Steps:
1. Adding other classification models for shorting and holding stocks instead of purchasing and selling every **_x_** minutes
2. Buying and selling neglected price to trade on specific platforms which is why stocks aren't held and are sold every **_x_** mintutes, so in the future this should be included
3. Extending this modeling to a higher frequency for 1 minute trading