# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path
from ml_trading.code.utils import mcnemar_test

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

# Import Data
df_predictions = pd.read_csv(full_path + "data/funds_predictions.csv", index_col=0)
df_results = pd.read_csv(full_path + "data/funds_results.csv", index_col=0)
print(df_results)

tickers = list(df_predictions.Ticker.unique())
models = list(df_predictions.Model.unique())




# McNemar's Test
df_predictions["Target"] = (df_predictions["Close"] > df_predictions["Open"]).replace({True: 1, False: 0})
df_predictions["Prediction"] = (df_predictions["seed_sum"] >= 5).replace({True: 1, False: 0})


for ticker in tickers:
    for model_1 in models:
        for model_2 in models:
            if model_1 != model_2:
                nemar = mcnemar_test(df_predictions, model_1, model_2, ticker)
                print(f"{ticker}, {model_1}, {model_2}: {nemar}")





