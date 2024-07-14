# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path
from ml_trading.code.utils import mcnemar_test

import pandas as pd
from rich import print


# Import Data
df_predictions = pd.read_csv(full_path + "data/funds_predictions_no_costs.csv", index_col=0)
df_results = pd.read_csv(full_path + "data/funds_results_no_costs.csv", index_col=0)
print(df_results)


tickers = list(df_predictions.Ticker.unique())
models = list(df_predictions.Model.unique())


df_predictions["Target"] = df_predictions["true_label"]
df_predictions["Prediction"] = df_predictions["Accuracy_calculation"]

print(df_predictions.head(n=10))


for ticker in tickers:
    for model_1 in models:
        for model_2 in models:
            if model_1 != model_2:
                nemar = mcnemar_test(df_predictions, model_1, model_2, ticker)
                print(f"{ticker}, {model_1}, {model_2}: {nemar}")





