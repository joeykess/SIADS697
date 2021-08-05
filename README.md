**<h2>SIADS697 Project</h2>**
* **<h3>Tyler Drumheller, Jeff Garden, Joseph Kessler</h3>**

<b>Goal:</b> Gathering NYSE historical data and utilizing it to create various portfolio management strategies
* Tyler's method utilizes short to medium term (7 to 120 day) technical analysis to predict prices over those terms using a Random Forest Regressor model
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


<H2>Financial Modeling Exploration Dashboard</H2>

<b>Goal</b>
The goal of our Financial Modeling Exploration Dashboard is to allow users to explore stock performance and model predictive power to help make assessments on which stocks they should buy. Our models can be used to build a portfolio autonomously, or used to help users manual select stocks.

<b>Implementation</b>
Our dashboard runs from python scripts using Plotly Dash. It is hosted on AWS as a containerized Docker image, leveraging AWS Elastic Container Service. It follows a similar structure to Dash dashboard implementations, in that it uses an app.py file to instantiate the dashboard, but "pages" are created separately and loaded together through python import statements. All pages are in our home driectory, but all data and CSS styling files are stored in the assets folder.

The Steps to Load the Dashboard Locally are as Follows:
1. Open the index.py file in a python IDE, such as Atom
2. Run this file in your IDE or from the terminal directly
3. Once running, the dashboard will load to a local server
4. Your terminal may automatically open the dashboard in your browser, otherwise copy the IP address printed out and paste in browser
5. You can make changes to individual pages directly, and saving will cause the dashboard to hot reload (automatically refresh)
6. Any errors in the python code that cause the dashboard to fail will need to be fixed, and then index.py needs to be re-run

