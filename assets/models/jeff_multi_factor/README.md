## Multifactor Multilayer Perceptron

#### Background

<p>Factor investing is a security selection approach that involves targeting specific elements or factors that appear to drive the returns of an asset. The premise behind Factor investing is that using quantitative methods, various risk premia can be identified isolated and utilized to the investors advantage. This idea is the corner stone behind almost all quantitative portfolio management methods. Factor investing is not a new idea, but recent developments in computing and mathematics (machine learning) allow practitioners to approach factor investing from an entirely different angle.</p>

#### Theoretical Framework: 
 
1) __Investing is about probabilities:__ More importantly, investing is about conditional probabilities, and the portfolio manager's primary task is to answer one simple question - "Given what I know right now, what are the odds security xxx moves in a desirable way?" 
2) __Portfolio Management is Bayesian in Nature:__ Portfolio managers and analysts are constantly bombarded with data in all forms. It is their job to process, analyze and update expectaions as information comes in.  
3) __Success is Measured in Relative Terms:__ Avoiding bad stocks is at least as important as buying good ones. Generating strong returns only adds value if we are taking reasonable risks to achieve our returns. As such buying a select few stocks with high probabilities of success and avoiding stocks with a low probability of success at all costs should result in strong performance.

#### Data:

<p> The model uses data from four different categories:</p>

1) Accounting data – this is data taken directly from a company’s financial statements (income statement, balance sheet, cashflow statement). 
2) Trading data – This includes data that is based on market activity over a given period of time. 
3) Valuation data – This is generally market data normalized by accounting data 
4) Technical Indicators – This includes moving averages for various windows based on the price of the stock.
5) The Label consists of a 1 for all stocks with a 1 year total return in the top 50% and a 0 for those in the bottom 50%.

<p> In total the model uses 69 features and one binary label<p/>

#### Model

<p> Below is a step by step guide to the model found in `Multi_factor_mlp.ipynb` file</p>

##### Data Preperation

1. __Data Preprocessing and cleaning:__ The data comes from multiple sources as raw numerical values recorded on a given day for a given asset. Missing data is discarded to minimize errors. Numerous features are engineered by combining two features (for example P/E ratios are calculated by normalizing daily prices by the most recent trailing 12 month EPS)  
2. __Data analysis and visualization__: Multiple visualizations are generated to help understand distributions, understand correlations, and identify any obvious relationships between features and themselves and features and the label.

##### Feature Preprocessing 

3. __Outlier detection and elimination:__ To eliminate outliers, we chose to use the isolation forest anomaly detection algorithm.
4. __Train Test Split:__ Data is split into training and testing sets (80% training, 20% test)
5. __Scaling:__ The data is normalized using a standard scaler, we use the training set only to fit the scaler, then scale the test data using the scaler that was fit on the training data.

##### Model Construction

6. __Model Tuning and Selection:__ We have used The Keras package to construct our model. We chose to build a sequential model with three hidden layers and multiple nodes, all of which use the 'relu' activation function, and the output layer with a single node that uses the sigmoid activation function. We use the binary cross entropy loss function and the adam optimizer. We also include dropouts after every hidden layer to help minimize potential overfitting and use l2 regularization to help with any effects we may experience because of multicollinearity. Lastly, we use the Keras Random Search tuner to identify the ideal number of nodes in each hidden layer, and the most effective learning rate for the optimizer. The tuner objective is set to minimize the loss function results for the validation (test) set. Hyperparameters for the best model are stored then applied to our production model.
7. __Model evaluation:__ We retrain a model using the hyperparameters identified by the tuner in the previous step. We evaluate the model based on Loss, Binary accuracy, Precision and Recall.

##### Simulation

8. __Pipline:__ The model is fed cleaned, scaled historic data that has gone through the same outlier removal process described previously. The entirety of the three month block is used to train a model with the same hyperparameters as previously identified. The resulting model is used to make a single prediction on all S&P 500 Stocks with available data using observations from the very next day. Predictions are stored and recorded for portfolio construction.
9. __Portfolio construction:__ All stocks are sorted based on the probability calculated by the model, the top 50 stocks are taken into consideration. These are then sorted by average market cap. All stocks with a weight of less than 2% of the sum of market cap for all stocks in consideration are eliminated. Weights are recalculated and sent into the trading engine for execution.
10. __Trading:__ The portfolio trades on the first trading day of every quarter. For the sake of simplicity the trading engine sells all open positions each trading day, then buys a new portfolio. The Trading engine uses a the port_2 class that stores daily snapshots of the all relevant portfolio information needed to calculate performance, execute trades, manage cash, and see historic and current positions.

##### Evaluation

11. __Preformace Evaluation:__ The model results results are evaluated using both absolute and riska adjusted metrics, as well as CAPM analysis.

#### Further Consideration

1. Evaluation at different rebalancing intervals (weekly, monthly, bi-monthly) 
2. Use of other intervals for features (Daily data may be noisy, weekly may be better)
3. Change the time frame of the label (testing both shorter and longer terms)
4. Addition and or elimination of features - there are still many data sets worth exploring that were not included in the analysis






