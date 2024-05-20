# Functions and classes that are executed repeatedly

import torch
from torch import nn
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.activation = nn.SiLU() # hyperparameter
        self.fcs = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        for i in range(self.num_layers):
            input_size = self.input_dim if i == 0 else self.hidden_dim
            fc = nn.Linear(input_size, self.hidden_dim)
            setattr(self, 'fc%i' % i, fc)
            self._set_init(fc)
            self.fcs.append(fc)

        self.predict = nn.Linear(self.hidden_dim, self.output_dim)
        self._set_init(self.predict)

    def _set_init(self, layer):
        nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        pre_activation = [x]
        layer_input = [x]

        for i in range(self.num_layers):
            x = self.fcs[i](x)
            pre_activation.append(x)

            x = self.activation(x)
            layer_input.append(x)

        out = self.predict(x)
        out = torch.sigmoid(out)   # Binary Softmax Activation Function

        return out, layer_input, pre_activation




def calculate_wma(length):
    weights = np.arange(1, length + 1)  # [1, 2, ..., length]
    def wma(series):
        return np.dot(series, weights) / weights.sum()

    return wma



def mcnemar_test(df_predictions, model_1, model_2, ticker):

    results = []
    subset_ticker = df_predictions[df_predictions["Ticker"] == ticker]

    pred_model_1 = subset_ticker[df_predictions['Model'] == model_1]["Prediction"]
    pred_model_2 = subset_ticker[df_predictions['Model'] == model_2]["Prediction"]

    if len(pred_model_1) != len(pred_model_2):
        raise ValueError(f"Predictions from both models must have the same number of samples for ticker {ticker}.")

    # Contingency table
    n_1_0 = 0
    n_0_1 = 0

    for p1, p2 in zip(pred_model_1, pred_model_2):
        if p1 == 1 and p2 == 0:
            n_1_0 += 1
        elif p1 == 0 and p2 == 1:
            n_0_1 += 1

    table = [[0, n_1_0], [n_0_1, 0]]

    result = mcnemar(table, exact=True)

    print(n_0_1)
    print(n_1_0)
    results.append({
        'Ticker': ticker,
        'Model1': pred_model_1,
        'Model2': pred_model_2,
        #'b': n_1_0,
        #'c': n_0_1,
        #'statistic': result.statistic,
        #'pvalue': result.pvalue
    })

    #return pd.DataFrame(results)
    return result.pvalue


