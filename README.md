**<h2>SIADS697 Project</h2>**
* **<h3>Tyler Drumheller, Jeff Garden, Joseph Kessler</h3>**

<b>Goal:</b> Gathering NYSE historical data and utilizing it to create various portfolio management strategies
* Tyler's method utilizes...
* Jeff's method utilizes...
* Joey's model utilizes short term trading trends read from candlestick charts through a CNN model to employ a high-risk/high-reward method
<br></br>
Multiple scripts will be utilized to pull raw data:
* `historical_stocks.py` will be utilized to get dailies from select symbols utilizing package yfinance
* `short_term_stocks.py` will be utilized to get intraday 5m interval utilizing AlphaVantage API
  * to run this code you must have an API key from AlphaVantage

Steps for creating data used for analysis:
1. From project home directory, run `historical_stocks.py` or `short_term_stocks.py`, which downloads daily closing data for all S&P 500 tickers or select high volume stocks not listed on the NASDAQ (AlphaVantage does not offer this data)
2. Load `fundamentals_spy.csv` to get details for 10Q and 10K earnings data
3. `short_chart_creator.py` to get training and validation data for neural network

Model creation and utilization:
1. Pick model type
   1. technical model utilizing CNNs for intraday trading
   2. technical model utilizing *regression* for weekly trading
   3. quantitative model utilizing *regression* for *monthly* trading
2. Train and fit models
   1. `short_cnn.py` to create intraday CNN model
   2. Tyler's model and data
   3. Jeff's model and data
3. Utilize `portfolio.py` or `intraday_portfolio.py` classes to trade stocks and track performance, trades, and other metrics

Resulting analysis:
...

