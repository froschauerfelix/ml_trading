# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path, features

import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from ast import literal_eval
from matplotlib import pyplot as plt
from rich import print


# Import the best parameters
df_hyperparameter = pd.read_csv(full_path + "data/funds_hyperparameter.csv", index_col=0)
funds_processed = pd.read_csv(full_path + "data/funds_data_processed.csv", index_col=0)

# Train the model once with all all training data
which_model = "svm"
print(df_hyperparameter)


#tickers = list(funds_processed.Ticker.unique())
tickers = ["IYE"]

ticker_dfs = []

for ticker in tickers:
    print(ticker)
    # Get the data
    ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

    ticker_train = ticker_return[ticker_return["Type"] == "train"]
    ticker_test = ticker_return[ticker_return["Type"] == "test"]

    X_train = ticker_train[features]
    Y_train = ticker_train["Target_tomorrow"]

    X_test = ticker_test[features]
    Y_test = ticker_test["Target_tomorrow"]

    # Random Forest
    if which_model == "rf":
        model_rn = "random_forest"

        # get the hyperparameter

        hyper = df_hyperparameter[(df_hyperparameter["Ticker"] == ticker) & (df_hyperparameter["Model"] == model_rn)].Parameter
        print(hyper)
        param1, param2, param3, param4 = hyper.iloc[0].split("[")[1].split("]")[0].split(",")
        print("testing")

        if param2 != " None":
            param2 = int(param2)
        else:
            param2 = None

        model_rf = RandomForestClassifier(n_estimators=int(param1),
                                          max_depth=param2,
                                          min_samples_split=int(param3),
                                          min_samples_leaf=int(param4),
                                          random_state=1)

        model_rf.fit(X_train, Y_train)
        test_score = model_rf.score(X_test, Y_test)
        predictions = model_rf.predict(X_test)
        predict_proba = model_rf.predict_proba(X_test)[:, 1]
        print("rf")
        print(model_rf.predict_proba(X_test))

        ticker_results = ticker_return[ticker_return["Type"] == "test"]
        ticker_results["Prediction"] = predictions
        ticker_results["Prediction_Probability"] = predict_proba

        ticker_dfs.append(ticker_results)

    if which_model == "svm":
        model_rn = "svm"

        # get the hyperparameter
        parameter_string = df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker)
                                                 & (df_hyperparameter['Model'] == 'svm'), 'Parameter'].iloc[0]
        parameters = literal_eval(parameter_string)

        # if Kernel: poly
        if len(parameters) == 4:

            model_svm = svm.SVC(C=parameters[0], kernel=parameters[1], gamma=parameters[2],
                                degree=parameters[3], probability=True).fit(X_train, Y_train)

            test_score = model_svm.score(X_test, Y_test)
            predictions = model_svm.predict(X_test)
            predict_proba = model_svm.predict_proba(X_test)[:, 1]
            print("svm")
            print(model_svm.predict_proba(X_test))


            ticker_results = ticker_return[ticker_return["Type"] == "test"]
            ticker_results["Prediction"] = predictions.copy()
            ticker_results["Prediction_Probability"] = predict_proba

            ticker_dfs.append(ticker_results)

        else:
            parameter_string = df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == 'svm'), 'Parameter'].iloc[0]
            parameters = literal_eval(parameter_string)


            model_svm = svm.SVC(C=parameters[0], kernel=parameters[1], gamma=parameters[2], probability=False).fit(X_train, Y_train)
            test_score = model_svm.score(X_test, Y_test)
            predictions = model_svm.predict(X_test)
            predict_proba = model_svm.predict_proba(X_test)[:, 1]


            ticker_results = ticker_return[ticker_return["Type"] == "test"]
            ticker_results["Prediction"] = predictions.copy()
            ticker_results["Prediction_Probability"] = predict_proba

            ticker_dfs.append(ticker_results)



# combine all ticker dataframes together
df_results = pd.concat(ticker_dfs)



#df_results = df_results.drop(["DP_1", "DP_2", "MA_5", "MA_10", "MA_30", "MPP_30","MPP_50", "DV_1", "DV_2", "Target_tomorrow", "Type"], axis=1)

df_results = df_results[["Ticker", 'Open', 'Close', 'Target', 'Prediction', 'Prediction_Probability']] #, "Prediction_Probability"]]

print(df_results.head(n=40))




#df_results["Signal"] = ['BUY' if x >= 0.55 else 'HOLD' if x >= 0.45 else 'SELL' for x in df_results.Prediction_Probability]
df_results["Signal"] = ['BUY' if x >= 0.5 else 'SELL' for x in df_results.Prediction_Probability]

df_results["Signal_unchanged"] = df_results["Signal"]



# Logic to implement the described behavior
change_to_hold = False


for i in range(len(df_results)):
    if df_results.iloc[i]['Signal'] == 'SELL':
        change_to_hold = False
    elif df_results.iloc[i]['Signal'] == 'BUY' and change_to_hold:
        df_results.iloc[i, df_results.columns.get_loc('Signal')] = 'HOLD'
    elif df_results.iloc[i]['Signal'] == 'BUY':
        change_to_hold = True


print(df_results.head(n=50))
print(df_results.columns)





# Signal dataframe

signal_df = df_results[(df_results["Signal"] == "SELL") | (df_results["Signal"] == "BUY")]

# Initialize variables
investment = 1
is_buy = False



investments = []
dates = []

for i in range(len(signal_df)):

    if signal_df.iloc[i]["Signal"] == "BUY":
        buy_price = signal_df.iloc[i]["Open"]
        is_buy = True

    elif signal_df.iloc[i]["Signal"] == "SELL" and is_buy:
        sell_price = signal_df.iloc[i]["Close"]
        return_rate = (sell_price - buy_price) / buy_price
        investment *= (1+ return_rate)

        print(signal_df.index[i])
        dates.append(signal_df.index[i])
        investments.append(investment)
        print(investment)
        is_buy = False



inv_return = (investment -1) / 1
print(f"Return S&P in RF: {round(inv_return*100, 4)}%")

# Alternative (buy and hold benchmark strategy)
first_open = signal_df.iloc[1].Open
last_close = signal_df.iloc[-1].Close
benchmark_return = (last_close - first_open) / first_open
print(f"Return S&P buy&hold: {round(benchmark_return*100, 4)}%")





# Plotting
df_results.index = pd.to_datetime(df_results.index)

plt.figure(figsize=(14, 7))
plt.plot(df_results.index, df_results['Close'], label='Close Price', color='skyblue')
plt.scatter(df_results[df_results['Signal'] == "SELL"].index, df_results[df_results['Signal'] == "SELL"]['Close'], label='Sell Signal', color='red', marker='^')
plt.scatter(df_results[df_results['Signal'] == "BUY"].index, df_results[df_results['Signal'] == "BUY"]['Close'], label='Buy Signal', color='green', marker='^')




plt.title('Stock Market Predictions with Buy Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()




############# JUNK ##############
"""
# Import the predictions
funds_predictions = pd.read_csv(full_path + "data/funds_predictions.csv", index_col=0)




trading_days = funds_predictions[funds_predictions["Signal"] == 0]
print(trading_days)

return_sum = trading_days["Daily_Return"].sum()
print(return_sum)


print("Benchmark")

first_open = funds_predictions.Open[1]
last_close = funds_predictions.Close[-1]
benchmark_return = (last_close - first_open) / first_open

print(benchmark_return)




funds_predictions.index = pd.to_datetime(funds_predictions.index)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(funds_predictions.index, funds_predictions['Close'], label='Close Price', color='skyblue')
plt.scatter(funds_predictions[funds_predictions['Signal'] == 1].index, funds_predictions[funds_predictions['Signal'] == 1]['Close'], label='Buy Signal', color='red', marker='^')

plt.title('Stock Market Predictions with Buy Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
"""

# Plotting


"""df_results.index = pd.to_datetime(df_results.index)


plt.figure(figsize=(14, 7))
plt.plot(df_results.index, df_results['Close'], label='Close Price', color='skyblue')
plt.scatter(df_results[df_results['Prediction'] == 1].index, df_results[df_results['Prediction'] == 1]['Close'], label='Buy Signal', color='green', marker='^')
plt.scatter(df_results[df_results['Prediction'] == 0].index, df_results[df_results['Prediction'] == 0]['Close'], label='Sell Signal', color='red', marker='^')



plt.title('Stock Market Predictions with Buy Signals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
"""
#plt.show()

