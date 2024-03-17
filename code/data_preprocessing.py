# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, train_end, validate_end

import pandas as pd
from rich import print
from sklearn.preprocessing import MinMaxScaler


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

    # Moving Averages (MA)
    moving_averages = [5, 10, 30]
    for move in moving_averages:
        col_name = f"MA_{move}"
        relevant_variables.append(col_name)
        # Scale the MA by the closing price
        ticker_return[col_name] = ticker_return["Close"].rolling(move).mean() / ticker_return["Close"]

    # Moving Price Level Percentage (MPP)
    moving_prices = [10, 30, 50]
    for price in moving_prices:
        col_name = f"MPP_{price}"
        relevant_variables.append(col_name)
        min_price = ticker_return["Low"].rolling(price).min()
        max_price = ticker_return["High"].rolling(price).max()
        ticker_return[col_name] = (ticker_return["Close"] - min_price) / (max_price - min_price).replace(0, pd.NA)

    # Percentage Price Oscillator
    #
    #
    #

    # Split the dataset into train and set test
    ticker_train = ticker_return[ticker_return.index < validate_end].copy()
    ticker_test = ticker_return[ticker_return.index >= validate_end].copy()

    # Scale the feature "Volume", the test set needs to be scaled with the data from the train set to avoid leakage
    volume_scaler = MinMaxScaler()
    volume_scaler.fit(ticker_train[["Volume"]])

    # Fit the scaler
    ticker_train[["Volume"]] = volume_scaler.transform(ticker_train[["Volume"]])
    ticker_test[["Volume"]] = volume_scaler.transform(ticker_test[["Volume"]])

    # Change of Volume for train and test data
    ticker_train["DV_1"] = ticker_train.Volume.diff()
    ticker_train["DV_2"] = ticker_train["DV_1"].shift(1)
    ticker_test["DV_1"] = ticker_test.Volume.diff()
    ticker_test["DV_2"] = ticker_test["DV_1"].shift(1)

    # Shift the Target Variable such that the data today predicts the target tomorrow
    ticker_train = ticker_train.assign(Target_tomorrow=ticker_train.Target.shift(-1))
    ticker_train = ticker_train.dropna()

    ticker_test = ticker_test.assign(Target_tomorrow=ticker_test.Target.shift(-1))
    ticker_test = ticker_test.dropna()

    # Kick out irrelevant columns
    ticker_train = ticker_train.drop(["High", "Low", "Adj Close", "Volume"], axis=1)
    ticker_test = ticker_test.drop(["High", "Low", "Adj Close", "Volume"], axis=1)

    # Combine again for faster data handling
    ticker_train["Type"] = "train"
    ticker_test["Type"] = "test"

    ticker_data = pd.concat([ticker_train, ticker_test], axis=0)
    ticker_dfs.append(ticker_data)


# combine all ticker dataframes together
processed_data = pd.concat(ticker_dfs)
processed_data.to_csv(full_path + "data/funds_data_processed.csv", encoding="utf-8", index=True)



print(processed_data)
print(processed_data.columns)

print('[green]The data is processed now. Now, the model is ready to be trained.[/]')

