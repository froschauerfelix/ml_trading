# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, validate_end, features
from ml_trading.code.utils import calculate_wma


import pandas as pd
from rich import print
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
import numpy as np

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
    relevant_variables = []


    # Simple Moving Averages (SMA)
    moving_averages = [15, 50]
    for move in moving_averages:
        col_name = f"SMA_{move}"
        relevant_variables.append(col_name)
        # Scale the MA by the closing price
        ticker_return[col_name] = ticker_return["Close"].rolling(move).mean()

    # Create the differences
    ticker_return["SMA_15_minus_SMA_50"] = ticker_return["SMA_15"] - ticker_return["SMA_50"]
    relevant_variables.append("SMA_15_minus_SMA_50")


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

    ticker_return['RSI'] = (100 - (100 / (1 + rs))) # /100 to normalize


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
    ticker_return["OBVMA_50"] = ticker_return["OBV"].rolling(50).mean()


    # Differences in OBV
    ticker_return['OBV_MA_diff'] = ticker_return['OBVMA'] - ticker_return['OBVMA_50']


    # Exponential Moving Averages (EMA)
    ema_periods = [12, 26]  # good for MACD
    for period in ema_periods:
        col_name = f"EMA_{period}"
        relevant_variables.append(col_name)
        ticker_return[col_name] = ticker_return["Close"].ewm(span=period, adjust=False).mean() #/ ticker_return["Close"]


    # Scale and split the data
    scaler = StandardScaler()

    # Split the dataset into train and set test
    ticker_train = ticker_return[ticker_return.index < validate_end].copy()
    ticker_test = ticker_return[ticker_return.index >= validate_end].copy()

    # Fit scaler to training data
    features = ["SMA_15_minus_SMA_50", "MACD_Line", "MACD_Histogram", "RSI", "OBV_MA_diff", "SMA_15", "SMA_50",
                "EMA_12", "EMA_26", "MACD_Signal", "OBV", "OBVMA"]


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



print(processed_data.head(n=5))
print(processed_data.columns)

print('[green]The data is processed now. Now, the model is ready to be trained.[/]')


"""
# Deleted Features

# Change of Price
ticker_return["DP_1"] = ticker_return.Close.diff() # yesterday to today
ticker_return["DP_2"] = ticker_return["DP_1"].shift(1) # the day before yesterday to yesterday

# Change of Volume
ticker_return["DV_1"] = ticker_return.Volume.diff() # yesterday to today
ticker_return["DV_2"] = ticker_return["DV_1"].shift(1) # the day before yesterday to yesterday

"""