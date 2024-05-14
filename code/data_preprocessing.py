# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, train_end, validate_end, features
from ml_trading.code.utils import calculate_wma


import pandas as pd
from rich import print
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)


# Import the generated data
funds_return = pd.read_csv(full_path + "data/funds_data_raw.csv", index_col=0)
tickers = list(funds_return.Ticker.unique())




# Define Variables for each Ticker symbol and combine the dataframe afterwards again
ticker_dfs = []
for ticker in tickers:
    print(ticker)
    # subset with the selected ticker
    ticker_return = funds_return[funds_return["Ticker"] == ticker].copy()

    # define the target variable
    ticker_return["Target"] = (ticker_return["Close"] > ticker_return["Open"]).replace({True: 1, False: 0})

    ### Define Independent Variables ###
    relevant_variables = ["DP_1", "DP_2", "DV_1", "DV_2"]


    # Change of Price
    ticker_return["DP_1"] = ticker_return.Close.diff() # yesterday to today
    ticker_return["DP_2"] = ticker_return["DP_1"].shift(1) # the day before yesterday to yesterday

    # Change of Volume
    ticker_return["DV_1"] = ticker_return.Volume.diff() # yesterday to today
    ticker_return["DV_2"] = ticker_return["DV_1"].shift(1) # the day before yesterday to yesterday



    # Simple Moving Averages (SMA)
    moving_averages = [10, 25, 50] # random
    for move in moving_averages:
        col_name = f"SMA_{move}"
        relevant_variables.append(col_name)
        # Scale the MA by the closing price
        ticker_return[col_name] = ticker_return["Close"].rolling(move).mean() / ticker_return["Close"]

    # Exponential Moving Averages (EMA)
    ema_periods = [12, 26]  # good for MACD
    for period in ema_periods:
        col_name = f"EMA_{period}"
        relevant_variables.append(col_name)
        ticker_return[col_name] = ticker_return["Close"].ewm(span=period, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD)
    ticker_return['MACD_Line'] = ticker_return['EMA_12'] - ticker_return['EMA_26']
    relevant_variables.append('MACD_Line')

    # Signal Line
    ticker_return['MACD_Signal'] = ticker_return['MACD_Line'].ewm(span=9, adjust=False).mean()
    relevant_variables.append('MACD_Signal')

    # MACD Histogram
    ticker_return['MACD_Histogram'] = ticker_return['MACD_Line'] - ticker_return['MACD_Signal']
    relevant_variables.append('MACD_Histogram')

    # Relative Strength Index (RSI)
    rsi_window = 14

    delta = ticker_return["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0)
    gain_ema = gain.ewm(span=rsi_window, adjust=False).mean()
    loss_ema = -loss.ewm(span=rsi_window, adjust=False).mean()
    rs = gain_ema / loss_ema

    ticker_return['RSI'] = (100 - (100 / (1 + rs))) / 100  # /100 to normalize


    # On-Balance Volume (OBV)
    ticker_return['OBV'] = 0
    ticker_return['OBV'].iloc[0] = ticker_return['Volume'].iloc[0].copy()

    for i in range(1, len(ticker_return)):
        # Check if the current close is higher than the previous close
        if ticker_return['Close'].iloc[i] > ticker_return['Close'].iloc[i - 1]:
            ticker_return['OBV'].iloc[i] = ticker_return['OBV'].iloc[i - 1] + ticker_return['Volume'].iloc[i]
        # Check if the current close is lower than the previous close
        elif ticker_return['Close'].iloc[i] < ticker_return['Close'].iloc[i - 1]:
            ticker_return['OBV'].iloc[i] = ticker_return['OBV'].iloc[i - 1] - ticker_return['Volume'].iloc[i]
        # If the current close is the same as the previous close
        else:
            ticker_return['OBV'].iloc[i] = ticker_return['OBV'].iloc[i - 1].copy()

    # OBV Moving Average (OBVMA)
    ticker_return["OBVMA"] = ticker_return["OBV"].rolling(10).mean() #/ #ticker_return["OBV"]

    # Exponential Moving Averages (EMA)
    ema_periods = [12, 26]  # good for MACD
    for period in ema_periods:
        col_name = f"EMA_{period}"
        relevant_variables.append(col_name)
        ticker_return[col_name] = ticker_return["Close"].ewm(span=period, adjust=False).mean() / ticker_return["Close"]


    # Scale and split the data
    scaler = StandardScaler()

    # Split the dataset into train and set test
    ticker_train = ticker_return[ticker_return.index < validate_end].copy()
    ticker_test = ticker_return[ticker_return.index >= validate_end].copy()

    # Fit scaler to training data
    features = ["OBV", "OBVMA"]
    scaler.fit(ticker_train[features])

    # Transform both training and test data
    ticker_train[features] = scaler.transform(ticker_train[features])
    ticker_test[features] = scaler.transform(ticker_test[features])

    # Shift the Target Variable such that the data today predicts the target tomorrow
    ticker_train = ticker_train.assign(Target_tomorrow=ticker_train.Target.shift(-1))
    ticker_train = ticker_train.dropna()

    ticker_test = ticker_test.assign(Target_tomorrow=ticker_test.Target.shift(-1))
    ticker_test = ticker_test.dropna()

    # Kick out irrelevant columns
    ticker_train = ticker_train.drop(["High", "Low", "Adj Close", "Volume"], axis=1)
    ticker_test = ticker_test.drop(["High", "Low", "Adj Close", "Volume"], axis=1)




    ticker_train["Type"] = "train"
    ticker_test["Type"] = "test"


    ticker_data = pd.concat([ticker_train, ticker_test], axis=0)
    ticker_dfs.append(ticker_data)


# combine all ticker dataframes together
processed_data = pd.concat(ticker_dfs)


processed_data.to_csv(full_path + "data/funds_data_processed.csv", encoding="utf-8", index=True)



print(processed_data.head(n=10))
print(processed_data.columns)

print('[green]The data is processed now. Now, the model is ready to be trained.[/]')

print(processed_data.head(n=10))


print(processed_data.EMA_12.describe())


# skipped features (junk)
"""
# Weighted Moving Averages (EMA)
weighted_averages = [5, 25]
for move in weighted_averages:
    col_name = f"WMA_{move}"
    relevant_variables.append(col_name)

    ticker_return[col_name] = ticker_return["Close"].rolling(move).apply(calculate_wma(move), raw=True) /  ticker_return["Close"]
"""


"""
# Moving Price Level Percentage (MPP)
moving_prices = [10, 30, 50]
for price in moving_prices:
    col_name = f"MPP_{price}"
    relevant_variables.append(col_name)
    min_price = ticker_return["Low"].rolling(price).min()
    max_price = ticker_return["High"].rolling(price).max()
    ticker_return[col_name] = (ticker_return["Close"] - min_price) / (max_price - min_price).replace(0, pd.NA)
"""



"""
# Stochastic Oscillator
k_window = 14
d_window = 3
# %K line
low_min = ticker_return["Low"].rolling(window=k_window).min()
high_max = ticker_return["High"].rolling(window=k_window).max()
ticker_return["%K"] = (((ticker_return["Close"] - low_min) / (high_max - low_min)) * 100) / 100 # /100 normalize
# %D Line
ticker_return["%D"] = (ticker_return["%K"].rolling(window=d_window).mean())
"""



