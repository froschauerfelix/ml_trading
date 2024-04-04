# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, train_end, models, features, number_epochs
from ml_trading.code.utils import MLP

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import precision_score, accuracy_score

from rich import print
import itertools

import torch
from torch import nn


# Select what models to train (config.py)
random_forest = models[0]
support_vector_machine = models[1]
neural_network = models[2]


# Import data
funds_processed = pd.read_csv(full_path + "data/funds_data_processed.csv", index_col=0)

# Define the looping variable
tickers = list(funds_processed.Ticker.unique())

### Define a dataframe for the results ###
number_models = random_forest+neural_network+support_vector_machine # 3 if all models are True
number_rows = number_models * len(tickers)

# define ticker column
ticker_column = [ticker for ticker in tickers for _ in range(number_models)]

# define model column
potential_models = ["random_forest", "svm", "neural_network"]
list_models = [string for bool_val, string in zip(models, potential_models) if bool_val]
model_column = list_models * len(tickers)

# define dataframe
df_hyperparameter = pd.DataFrame(zip(ticker_column, model_column), columns=["Ticker", "Model"])
df_hyperparameter["Validation_Accuracy"] = np.nan
df_hyperparameter["Validation_Precision"] = np.nan
df_hyperparameter["Hyperparameter"] = np.nan
df_hyperparameter["% of buy signal"] = np.nan


for ticker in tickers:

    ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

    ticker_train = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index < train_end)]
    ticker_validate = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index > train_end)]

    # to tune the hyperparameters
    X_train = ticker_train[features]
    Y_train = ticker_train["Target_tomorrow"]

    X_validate = ticker_validate[features]
    Y_validate = ticker_validate["Target_tomorrow"]

    #print("Imbalance in validation set")
    #print(Y_validate.value_counts())
    ###################### RANDOM FOREST ######################
    if random_forest:
        model_rn = "random_forest"
        print(f"{ticker} is trained using a random forest.")

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [20, 100, 200],  # Number of trees in the forest
            'max_depth': [10, 20, None],  # Maximum number of levels in tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
            'min_samples_leaf': [2, 4]  # Minimum number of samples required at each leaf node
        }

        param_combinations = itertools.product(param_grid["n_estimators"],
                                               param_grid["max_depth"],
                                               param_grid["min_samples_split"],
                                               param_grid["min_samples_leaf"])

        best_validation_score = 0
        precision_of_best_validation_score = 0
        percentage_of_best_validation_score = 0
        for n_estimators, max_depth, min_samples_split, min_samples_leaf in param_combinations:

            model_rf = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              random_state=1)
            model_rf.fit(X_train, Y_train)
            Y_pred = model_rf.predict(X_validate)
            validation_accuracy = accuracy_score(Y_validate, Y_pred)
            validation_precision = precision_score(Y_validate, Y_pred, zero_division=0)
            buy_percentage = (sum(Y_pred) / len(Y_pred)) * 100


            # Save the parameters of the best validation score
            if validation_accuracy > best_validation_score:
                best_validation_score = validation_accuracy
                precision_of_best_validation_score = validation_precision
                percentage_of_best_validation_score = buy_percentage
                best_params = [n_estimators, max_depth, min_samples_split, min_samples_leaf]

        # Save the results
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Accuracy'] = best_validation_score

        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Precision'] = precision_of_best_validation_score

        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), '% of buy signal'] = percentage_of_best_validation_score

        params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Hyperparameter'] = params_as_string

    ################## SUPPORT VECTOR MACHINE #################
    if support_vector_machine:
        model_rn = "svm"
        print(f"{ticker} is trained using SVM.")

        # LINEAR, RBF, SIGMOID - Kernel
        param_grid_kernel = {
            'C': [0.1, 1, 10], # 100
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.0001, 'scale', 'auto'],
        }
        param_combinations_kernel = itertools.product(
            param_grid_kernel['C'],
            param_grid_kernel['kernel'],
            param_grid_kernel['gamma']
        )

        best_validation_score = 0
        precision_of_best_validation_score = 0
        percentage_of_best_validation_score = 0
        for C, kernel, gamma in param_combinations_kernel:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma, random_state=1).fit(X_train, Y_train)
            Y_pred = model_svm.predict(X_validate)
            validation_accuracy = accuracy_score(Y_validate, Y_pred)
            validation_precision = precision_score(Y_validate, Y_pred, zero_division=0)
            buy_percentage = (sum(Y_pred) / len(Y_pred)) * 100

            # Save the parameters of the best validation score
            if validation_accuracy > best_validation_score:
                best_validation_score = validation_accuracy
                precision_of_best_validation_score = validation_precision
                percentage_of_best_validation_score = buy_percentage
                best_params = [C, kernel, gamma]

        # POLY - Kernel
        param_grid_poly = {
            'C': [0.1, 1, 10], # 100
            'kernel': ['poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]
        }
        param_combinations_poly = itertools.product(
            param_grid_poly['C'],
            param_grid_poly['kernel'],
            param_grid_poly['gamma'],
            param_grid_poly['degree'],
        )

        for C, kernel, gamma, degree  in param_combinations_poly:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=1).fit(X_train, Y_train)
            Y_pred = model_svm.predict(X_validate)
            validation_accuracy = accuracy_score(Y_validate, Y_pred)
            validation_precision = precision_score(Y_validate, Y_pred, zero_division=0)
            buy_percentage = (sum(Y_pred) / len(Y_pred)) * 100

            # Save the parameters of the best validation score
            if validation_accuracy > best_validation_score:
                best_validation_score = validation_accuracy
                precision_of_best_validation_score = validation_precision
                percentage_of_best_validation_score = buy_percentage
                best_params = [C, kernel, gamma, degree]


        # Save the results
        if len(best_params) == 4: # Kernel: "poly"
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Accuracy'] = best_validation_score

            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Precision'] = precision_of_best_validation_score

            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), '% of buy signal'] = percentage_of_best_validation_score

            params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Hyperparameter'] = params_as_string


        # if kernel: linear, rbf, sigmoid
        else:
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Accuracy'] = best_validation_score

            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Validation_Precision'] = precision_of_best_validation_score

            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), '% of buy signal'] = percentage_of_best_validation_score

            params_as_string = str([best_params[0], best_params[1], best_params[2]])
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                    df_hyperparameter['Model'] == model_rn), 'Hyperparameter'] = params_as_string

    ###################### NEURAL NETWORK #####################
    if neural_network:
        model_rn = "neural_network"
        print(f"{ticker} is trained using a neural network.")

        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float)
        Y_train = torch.tensor(Y_train.to_numpy(), dtype=torch.long)

        X_validate = torch.tensor(X_validate.to_numpy(), dtype=torch.float)
        Y_validate = torch.tensor(Y_validate.to_numpy(), dtype=torch.long)


        # hyperparameter (config.py)
        number_epochs = number_epochs

        # Hyperparameter tuning
        param_grid = {
            'learning_rates': [10, 1,0.1, 0.01, 0.001, 0.0001],  # Number of trees in the forest
            'number_hidden': [1, 5, 10, 20],  # Maximum number of levels in tree
            'hidden_dim': [1, 2, 4, 8, 16]  # Minimum number of samples required to split a node
        }

        param_combinations = itertools.product(param_grid["learning_rates"],
                                               param_grid["number_hidden"],
                                               param_grid["hidden_dim"])

        best_validation_score = 0
        precision_of_best_validation_score = 0
        percentage_of_best_validation_score = 0
        for learning_rate, number_hidden, hidden_dim in param_combinations:

            # Model structure
            model_neural = MLP(len(features), 2, hidden_dim, number_hidden) # input, output, hidden_dim, #layers
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.NAdam(model_neural.parameters(), lr=learning_rate)

            for epoch in range(number_epochs):
                train_correct = 0
                train_loss = 0

                # Train
                model_neural.train()

                y_pred = model_neural(X_train)[0]
                loss = criterion(y_pred, Y_train)
                loss.backward()
                optimizer.step()

                # if last epoch
                if epoch == number_epochs-1:
                    model_neural.eval()
                    running_score = 0

                    with torch.no_grad():
                        y_pred = model_neural(X_validate)[0]
                        loss = criterion(y_pred, Y_validate)

                        _, index = torch.max(y_pred, 1)
                        running_score += torch.sum(index == Y_validate).item()

                    epoch_score = running_score / len(X_validate)
                    predicted_labels_np = index.reshape(-1, 1).numpy() #y_pred
                    Y_validate_np = Y_validate.reshape(-1, 1).numpy()

                    buy_percentage = ((sum(index) / len(index)) * 100).item()
                    validation_precision = precision_score(Y_validate_np, predicted_labels_np, zero_division=0)


                    # Save the parameters of the best validation score
                    if epoch_score > best_validation_score:
                        best_validation_score = epoch_score
                        precision_of_best_validation_score = validation_precision
                        percentage_of_best_validation_score = buy_percentage
                        best_params = [learning_rate, number_hidden, hidden_dim]

                    #print(f"Validation accuracy: {epoch_score}")


        # Save the results
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                df_hyperparameter['Model'] == model_rn), 'Validation_Accuracy'] = best_validation_score

        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                df_hyperparameter['Model'] == model_rn), 'Validation_Precision'] = precision_of_best_validation_score

        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                df_hyperparameter['Model'] == model_rn), '% of buy signal'] = percentage_of_best_validation_score

        params_as_string = str([best_params[0], best_params[1], best_params[2]])
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (
                df_hyperparameter['Model'] == model_rn), 'Hyperparameter'] = params_as_string



print(df_hyperparameter)

df_hyperparameter.to_csv(full_path + "data/funds_hyperparameter_all.csv", encoding="utf-8", index=True)