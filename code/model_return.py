# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, validate_end
import pandas as pd
from rich import print
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier



# Import the best parameters
funds_results = pd.read_csv(full_path + "data/funds_results.csv", index_col=0)
funds_processed = pd.read_csv(full_path + "data/funds_processed.csv", index_col=0)

# Train the model once with all all training data


print(funds_results)

tickers = list(funds_processed.Ticker.unique())


for ticker in tickers:
    print(ticker)
    # Get the data
    ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

    ticker_train = ticker_return[ticker_return["Type"] == "train"]
    ticker_test = ticker_return[ticker_return["Type"] == "test"]

    X_train = ticker_train.drop(["Ticker", "Target_tomorrow", "Type"], axis=1)
    Y_train = ticker_train["Target_tomorrow"]

    X_test = ticker_test.drop(["Ticker", "Target_tomorrow", "Type"], axis=1)
    Y_test = ticker_test["Target_tomorrow"]

    # Random Forest
    model_rn = "random_forest"

    # get the hyperparameter

    hyper = funds_results[(funds_results["Ticker"] == ticker) & (funds_results["Model"] == model_rn)].Parameter
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
    print(test_score)







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

plt.show()"""