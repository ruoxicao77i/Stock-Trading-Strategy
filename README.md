# Stock-Trading-Strategy

## Project Description

The objective of this project is to construct a short-term stock price prediction model based on historical stock price data and options market data, aiming to forecast the stock price trend for the next trading day. This project will utilize stock trading information from the previous trading day, including opening price, closing price, highest price, lowest price, trading volume, etc., as well as options market-related data, such as implied volatility, options trading volume, put/call ratio, etc., as the main input features.
In terms of methodology, the project will utilize statistical analysis and machine learning models to establish a correlation model between stock prices and option data, and predict the direction or magnitude of stock price changes on the next trading day. Based on the prediction results, corresponding trading strategies will be further designed, and historical data will be used to backtest their actual performance.

## Clear goals

The primary objective of this project is to utilize stock price trends from the previous trading day and option market data to successfully predict stock price movements for the next trading day, and to construct a trading model that outperforms the buy-and-hold strategy in historical backtesting.
Specific objectives include:
1. Construct a feature system encompassing stock trading data and options market indicators;
2. Train a predictive model to determine the trend or magnitude of stock price changes on the next trading day;
3. Design an executable trading strategy based on the prediction results;
4. Through historical backtesting, compare the performance of this strategy in terms of return and risk with that of the buy-and-hold strategy to verify its effectiveness.

## Data Collection

### Features

**For this project, we wish to collect the stock information with following features:**

- Stock price (Bid, Ask, Close, Volume, Open_Interest)
- Date
- Strike_Price
- Expiry_Date 
- Call/Put_Flag
- Greeks & IV
- Implied_Volatility 
- Delta 
- Gamma 
- Vega 
- Theta 

**The data collected above is all about the stock data but not have the information about the market data such as the interest rate and the VIX. So we wish to also collect the following features:**

- Benchmark Indices
- Volatility Indices
- Interest Rates
### Data Source

All the data above can be found in the WRDS (Wharton Research Data Services)(https://wrds-www.wharton.upenn.edu/). It contains the OptionMetrics database, which is industrial-level historical option data. We can use the school email to register an account and download the data in CSV format. 

## Modeling Plan

We want to predict next-day stock price movement. Hence, we will use regression task (ex: Next_Day_Return = (Close_t+1 − Close_t) / Close_t) and classification task (ex: Direction = sign(Next_Day_Return)).

### Baseline models
These models are easy to interpret and provide the benchmark for complex models.
- Linear Regression (predict next-day return)
- Logistic Regression (predict up or down movement)

### Model evaluation
For regression:
- Mean Squared Error (MSE)
- R^2 score

For classification:
- Accuracy
- Precision and Recall

### Simple trading strategy
We might design a model to connect predictions with making the dicision to tell user whether can buy the stock. We will implement a simple rule-based strategy.
- Buy when predicted return > 0
- Hold cash otherwise
We will compare this strategy with buy and hold using history data.


## Data Visualization Plan
We plan to use the following **tools** to visualize the data:

- Matplotlib
- Seaborn
- Plotly

We are thinking of using the following **charts** to visualize the data:

- **Standard OHLC + Volume:** to show the Candlestick Chart and Volume Bars.
- **Dual-Axis Line Chart:** to visualize the Stock Price and Implied Volatility. In this way, we can circle those moments when stock prices plummeted and IV skyrocketed.
- **Correlation Heatmap:** Put your features (IV, Gamma, PCR, VIX) and target variable (Next_Day_Return) together to see the correlation between them.

## Test Plan
Because this project uses time-series financial data, we will split the dataset in timeline rather than randomly.

For example:
- Training data: first 80% of the timeline
- Testing data: last 20% of the timeline
This ensures that the model only uses past information to predict future stock.

Model evaluation:
- for regression: Mean Squared Error
- for classification: Accuracy





