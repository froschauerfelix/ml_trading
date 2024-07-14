# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path, features, number_epochs, with_costs, average_costs_etf
from ml_trading.code.utils import MLP

import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, accuracy_score
from torch.nn.functional import softmax

import random
from ast import literal_eval
from matplotlib import pyplot as plt
from rich import print

import torch
from torch import nn

# Import the best parameters
df_hyperparameter = pd.read_csv(full_path + "data/funds_hyperparameter_01.csv", index_col=0)
funds_processed = pd.read_csv(full_path + "data/funds_data_processed.csv", index_col=0)

# All models that are selected in the config file
models = df_hyperparameter.Model.unique()

# Select all tickers for the loop
tickers = list(df_hyperparameter.Ticker.unique())

# number seeds
num_seeds = 10

# Define results dataframe
df_results = df_hyperparameter[["Ticker", "Model"]].copy()

# delivers the accuracies for all 10 seeds
df_results["Test_Accuracy"] = np.nan
df_results["Test_Precision"] = np.nan


df_results["Imbalance_Predictions"] = np.nan
df_results["Imbalance_True"] = np.nan
df_results["Return_Model"] = np.nan
df_results["Return_Benchmark"] = np.nan
df_results["Number_Trades"] = np.nan
df_results["Test_Accuracy_new"] = np.nan
df_results["Number_buy"] = np.nan
df_results["Number_sell"] = np.nan



# dataframe for predictions per seed and per model/ticker
dfs_predictions = []

# dataframe to store evaluations per seed and per model/ticker
df_accuracy = df_hyperparameter[["Ticker", "Model"]].copy()
df_precision = df_hyperparameter[["Ticker", "Model"]].copy()
df_imbalance = df_hyperparameter[["Ticker", "Model"]].copy()
list_imbalance_true = []

# loop over all seeds
    # loop over all tickers
        # loop over all models
for seed in range(1, num_seeds+1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Current seed: {seed}")

    # name of column
    seed_col_a = f"accuracy_{seed}"
    seed_col_p = f"precision_{seed}"
    seed_col_i = f"prediction_imbalance_{seed}"
    seed_num = f"seed{seed}"


    # Get the final test score + Daily Predictions
    for ticker in tickers:
        print(f"Ticker: {ticker}")

        # Get the data
        ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

        ticker_train = ticker_return[ticker_return["Type"] == "train"]
        ticker_test = ticker_return[ticker_return["Type"] == "test"]

        X_train = ticker_train[features]
        Y_train = ticker_train["Target_tomorrow"]

        X_test = ticker_test[features]
        Y_test = ticker_test["Target_tomorrow"]

        # Calculate imbalance in test_set (append 3 times makes it easier to fit the results dataframe)
        imb_true = (sum(Y_test) / len(Y_test)) * 100
        list_imbalance_true.append(imb_true)
        list_imbalance_true.append(imb_true)
        list_imbalance_true.append(imb_true)



        # Random Forest
        if "random_forest" in models:
            model_rn = "random_forest"

            # get the hyperparameter
            hyper = df_hyperparameter[(df_hyperparameter["Ticker"] == ticker) & (df_hyperparameter["Model"] == model_rn)].Hyperparameter.values[0]
            parameters = literal_eval(hyper)

            # define the model with the best hyperparameters
            model_rf = RandomForestClassifier(n_estimators=parameters[0],
                                              criterion=parameters[1],
                                              max_depth=parameters[2],
                                              min_samples_split=parameters[3],
                                              min_samples_leaf=parameters[4])

            # train and predict
            model_rf.fit(X_train, Y_train)
            Y_pred = model_rf.predict(X_test)

            feature_importances = model_rf.feature_importances_

            test_accuracy = accuracy_score(Y_test, Y_pred)
            test_precision = precision_score(Y_test, Y_pred, zero_division=0)
            imbalance = (sum(Y_pred) / len(Y_pred)) * 100
            predict_proba = model_rf.predict_proba(X_test)[:, 1]

            # save the evaluation metrics for each seed iteration
            df_accuracy.loc[(df_accuracy['Ticker'] == ticker) & (
                    df_accuracy['Model'] == model_rn), seed_col_a] = test_accuracy
            df_precision.loc[(df_precision['Ticker'] == ticker) & (
                    df_precision['Model'] == model_rn), seed_col_p] = test_precision
            df_imbalance.loc[(df_imbalance['Ticker'] == ticker) & (
                    df_imbalance['Model'] == model_rn), seed_col_i] = imbalance


            # save the predictions
            ticker_results = ticker_test[["Ticker", "Open", "Close", "Target"]].copy()
            ticker_results["Seed"] = seed_num
            ticker_results["Model"] = model_rn
            ticker_results["Prediction"] = Y_pred

            #ticker_results["Prediction_Probability"] = predict_proba
            dfs_predictions.append(ticker_results)


        if "svm" in models:
            model_rn = "svm"

            # get the hyperparameter
            hyper = df_hyperparameter[(df_hyperparameter["Ticker"] == ticker) & (df_hyperparameter["Model"] == model_rn)].Hyperparameter.values[0]
            parameters = literal_eval(hyper)

            # if Kernel: poly
            if len(parameters) == 4:

                # define the model with the best hyperparameters
                model_svm = svm.SVC(C=parameters[0], kernel=parameters[1], gamma=parameters[2],
                                    degree=parameters[3], probability=True)
            else:
                model_svm = svm.SVC(C=parameters[0], kernel=parameters[1], gamma=parameters[2],
                                    probability=True)

            # train and predict
            model_svm.fit(X_train, Y_train)
            Y_pred = model_svm.predict(X_test)
            predict_proba = model_svm.predict_proba(X_test)[:, 1]

            test_accuracy = accuracy_score(Y_test, Y_pred)
            test_precision = precision_score(Y_test, Y_pred, zero_division=0)
            imbalance = (sum(Y_pred) / len(Y_pred)) * 100

            # save the accuracy for each seed iteration
            df_accuracy.loc[(df_accuracy['Ticker'] == ticker) & (
                    df_accuracy['Model'] == model_rn), seed_col_a] = test_accuracy
            df_precision.loc[(df_precision['Ticker'] == ticker) & (
                    df_precision['Model'] == model_rn), seed_col_p] = test_precision
            df_imbalance.loc[(df_imbalance['Ticker'] == ticker) & (
                    df_imbalance['Model'] == model_rn), seed_col_i] = imbalance

            # save the predictions
            ticker_results = ticker_test[["Ticker", "Open", "Close", "Target"]].copy()
            ticker_results["Seed"] = seed_num
            ticker_results["Model"] = model_rn
            ticker_results["Prediction"] = Y_pred

            #ticker_results["Prediction_Probability"] = predict_proba
            dfs_predictions.append(ticker_results)


        if "neural_network" in models:
            model_rn = "neural_network"

            # transform numpy to tensor
            X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float)
            Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.long)

            X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float)
            Y_test = torch.tensor(Y_test.to_numpy(), dtype=torch.long)

            # number of epochs (config.py)
            number_epochs = number_epochs

            # get the hyperparameter
            hyper = df_hyperparameter[(df_hyperparameter["Ticker"] == ticker) & (df_hyperparameter["Model"] == model_rn)].Hyperparameter.values[0]
            parameters = literal_eval(hyper)

            # define the model with the best hyperparameters
            learning_rate, number_hidden, hidden_dim = parameters

            model_neural = MLP(len(features), 2, hidden_dim, number_hidden)  # input, output, hidden_dim, #layers
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.NAdam(model_neural.parameters(), lr=learning_rate)

            # training
            for epoch in range(number_epochs):
                train_correct = 0
                train_loss = 0

                # Train
                model_neural.train()

                y_pred = model_neural(X_train)[0]
                loss = criterion(y_pred, Y_train)
                loss.backward()
                optimizer.step()

            # prediction
            model_neural.eval()

            running_score = 0
            with torch.no_grad():
                y_pred = model_neural(X_test)[0]
                loss = criterion(y_pred, Y_test)

                _, index = torch.max(y_pred, 1)
                running_score += torch.sum(index == Y_test).item()

            test_accuracy = running_score / len(X_test)
            predicted_labels_np = index.reshape(-1, 1).numpy()  # y_pred
            Y_test_np = Y_test.reshape(-1, 1).numpy()

            imbalance = ((sum(index) / len(index)) * 100).item()
            test_precision = precision_score(Y_test_np, predicted_labels_np, zero_division=0)

            probabilities = softmax(y_pred, dim=1)
            class_1_probabilities = probabilities[:, 1]

            # save the accuracy for each seed iteration
            df_accuracy.loc[(df_accuracy['Ticker'] == ticker) & (
                    df_accuracy['Model'] == model_rn), seed_col_a] = test_accuracy
            df_precision.loc[(df_precision['Ticker'] == ticker) & (
                    df_precision['Model'] == model_rn), seed_col_p] = test_precision
            df_imbalance.loc[(df_imbalance['Ticker'] == ticker) & (
                    df_imbalance['Model'] == model_rn), seed_col_i] = imbalance


            # save the predictions
            ticker_results = ticker_test[["Ticker", "Open", "Close", "Target"]].copy()
            ticker_results["Seed"] = seed_num
            ticker_results["Model"] = model_rn
            ticker_results["Prediction"] = predicted_labels_np
            #ticker_results["Prediction_Probability"] = class_1_probabilities.numpy()

            dfs_predictions.append(ticker_results)





# Define the mean accuracy and add it to results dataframe (this one is still wrong)
accuracy_columns = [col for col in df_accuracy.columns if col.startswith('accuracy')]
df_accuracy['accuracy_mean'] = df_accuracy[accuracy_columns].mean(axis=1)
df_results["Test_Accuracy"] = df_accuracy['accuracy_mean']




# Define the mean precision and add it to the results dataframe
precision_columns = [col for col in df_precision.columns if col.startswith('precision')]
df_precision['precision_mean'] = df_precision[precision_columns].mean(axis=1)
df_results["Test_Precision"] = df_precision['precision_mean']

# Define the mean imbalance predictions and add it to the results dataframe
imbalance_columns = [col for col in df_imbalance.columns if col.startswith('prediction')]
df_imbalance['imbalance_mean'] = df_imbalance[imbalance_columns].mean(axis=1)
print(df_imbalance)
df_results["Imbalance_Predictions"] = df_imbalance['imbalance_mean']

# Define the imbalance in the test set:
list_imbalance_true_short = list_imbalance_true[0:len(models)*len(tickers)]
df_results["Imbalance_True"] = list_imbalance_true_short



# combine all ticker dataframes together
df_predictions = pd.concat(dfs_predictions)


# change from long to wide layout
aggregated_df = df_predictions.groupby(['Date', 'Model', 'Ticker', 'Seed', 'Open', 'Close'])['Prediction'].mean().reset_index()
df_predictions = aggregated_df.pivot_table(index=['Date', 'Model', 'Ticker', 'Open', 'Close'],
                                    columns='Seed',
                                    values='Prediction')
df_predictions.reset_index(inplace=True)

# Sum all seed columns to get the majority vote
seed_columns = [col for col in df_predictions.columns if col.startswith('seed')]
df_predictions['seed_sum'] = df_predictions[seed_columns].sum(axis=1)


######################## RETURN CALCULATION ################################

# Transform the Prediction into a Trading Signal
threshold_buy = (num_seeds/2)
df_predictions["Signal"] = ['BUY' if x >= threshold_buy else 'SELL' for x in df_predictions.seed_sum]



# Calculate the alternative strategy (Benchmark: Buy and Hold)
for ticker in tickers:
    subset = df_predictions[(df_predictions["Ticker"] == ticker) & (df_predictions["Model"] == models[0])]
    first_open = subset.iloc[0].Open
    last_close = subset.iloc[-1].Close
    benchmark_return = (last_close - first_open) / first_open
    # add to the results
    df_results.loc[df_results['Ticker'] == ticker, 'Return_Benchmark'] = benchmark_return


# Transform Trade Signals into Return (two versions with and without the implementation of the costs

df_predictions["current_return"] = np.nan
df_predictions["number_trades"] = np.nan
combine_again = []



for ticker in tickers:
    # average spread for this ticker (/100 to adjust for percentage numbers)
    etf_cost = average_costs_etf[ticker]/100 if with_costs else 0

    for model in models:

        # Add a third class: HOLD (when a repeated signal occurs)
        subset = df_predictions[(df_predictions["Ticker"] == ticker) & (df_predictions["Model"] == model)].copy()

        #print(subset.head(n=10))
        subset["Accuracy_calculation"] = subset.Signal.shift(1).replace({"BUY": 1, "SELL": 0})
        subset["true_label"] = (subset["Close"] > subset["Open"]).replace({True: 1, False: 0})


        # Change repeated buy and sell signals into holds (no trading activity)
        subset['Signal'] = subset['Signal'].mask(subset['Signal'].eq('BUY') & subset['Signal'].shift().eq('BUY'), 'HOLD').copy()
        subset['Signal'] = subset['Signal'].mask(subset['Signal'].eq('SELL') & subset['Signal'].shift().eq('SELL'), 'HOLD').copy()

        # this shift is very important, to use the today's signal for tomorrow's trades
        # subset["signal"] shows the prediction for the next day, the shifted variation can be traded the same day
        subset["Adjusted_Signal"] = subset.Signal.shift(1)

        # sell everything on the last day
        subset.iloc[-1, subset.columns.get_loc("Adjusted_Signal")] = "SELL"

        # Implementation of the trading logic
        investment = 1
        is_buy = False
        adjusted_buy_price = 0
        number_trades = 0

        for i in range(len(subset)):


            if subset.iloc[i]["Adjusted_Signal"] == "BUY":
                buy_price = subset.iloc[i]["Open"]
                adjusted_buy_price = buy_price * (1+etf_cost)
                is_buy = True
                return_rate_b = 0

                subset.iat[i, subset.columns.get_loc('current_return')] = investment

                number_trades += 1
                subset.iat[i, subset.columns.get_loc('number_trades')] = number_trades


            elif subset.iloc[i]["Adjusted_Signal"] == "SELL" and is_buy:

                sell_price = subset.iloc[i]["Open"] # sell Open price
                adjusted_sell_price = sell_price # * (1-spread_cost_percent) (we just need to deduct the costs of the spread once)

                return_rate = (adjusted_sell_price - adjusted_buy_price) / adjusted_buy_price
                investment *= (1+return_rate)

                subset.iat[i, subset.columns.get_loc('current_return')] = investment
                is_buy = False

            else:
                if is_buy: # change the current_return daily because we hold the asset
                    current_price = subset.iloc[i]["Open"]
                    current_return = (current_price -adjusted_buy_price) / adjusted_buy_price
                    subset.iat[i, subset.columns.get_loc("current_return")] = investment * (1+ current_return)

                else: # not holding the asset right now
                    subset.iat[i, subset.columns.get_loc("current_return")] = investment


        # add the results to the correct row in the results dataframe
        df_results.loc[(df_results['Ticker'] == ticker) & (
                df_results['Model'] == model), 'Return_Model'] = (investment - 1) / 1

        df_results.loc[(df_results['Ticker'] == ticker) & (
                df_results['Model'] == model), 'Number_Trades'] = number_trades

        df_results.loc[(df_results['Ticker'] == ticker) & (
                df_results['Model'] == model), 'Test_Accuracy_new'] = (subset['Accuracy_calculation'] == subset['true_label']).mean()

        df_results.loc[(df_results['Ticker'] == ticker) & (
                df_results['Model'] == model), 'Number_buy'] = (subset['Accuracy_calculation'] == 1.0).sum()

        df_results.loc[(df_results['Ticker'] == ticker) & (
                df_results['Model'] == model), 'Number_sell'] = (subset['Accuracy_calculation'] == 0.0).sum()



        combine_again.append(subset)




df_predictions = pd.concat(combine_again)
df_results["beat"] = df_results.Return_Model > df_results.Return_Benchmark


print(df_results)


df_results.to_csv(full_path + "data/funds_results_with_costs.csv", encoding="utf-8", index=True)
df_predictions.to_csv(full_path + "data/funds_predictions_with_costs.csv", encoding="utf-8", index=True)

