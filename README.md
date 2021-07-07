# SIADS697

### Gathering DJI Stock historical data

Multiple scripts will be utilized to pull raw data:
- `historical_stocks.py` will be utilized to get dailies from select symbols
- `edgar_`*xx*`.py` will be utilized to draw information from SEC Edgar database



###Steps for creating data used for analysis:
1. From project home directory, run historical_stocks.py, which downloads daily closing data for the top 100 S&P 500 tickers
2. Load fundamentals_spy.csv to get details for 10Q and 10K earnings data
3. TBD...
