# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, train_end, models, features

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from rich import print
import itertools



from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import tree


# Select what models to train (config.py)
random_forest = models[0]
support_vector_machine = models[1]
neural_network = models[2]


# Import data
funds_processed = pd.read_csv(full_path + "data/funds_data_processed.csv", index_col=0)

#funds_return = pd.read_csv(full_path + "data/funds_return.csv", index_col=0)
tickers = list(funds_processed.Ticker.unique())


### Define a dataframe for the results
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
df_hyperparameter["Test_score"] = np.nan
df_hyperparameter["Parameter"] = np.nan



for ticker in tickers:

    ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

    ticker_train = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index < train_end)]
    ticker_validate = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index > train_end)]

    ticker_train_full = ticker_return[ticker_return["Type"] == "train"]
    ticker_test = ticker_return[ticker_return["Type"] == "test"]


    # to tune the hyperparameters
    X_train = ticker_train[features]
    Y_train = ticker_train["Target_tomorrow"]

    X_validate = ticker_validate[features]
    Y_validate = ticker_validate["Target_tomorrow"]

    # for getting a test score (Accuracy) and Final Predictions
    X_train_full = ticker_train_full[features]
    Y_train_full = ticker_train_full["Target_tomorrow"]

    X_test = ticker_test[features]
    Y_test = ticker_test["Target_tomorrow"]

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
        for n_estimators, max_depth, min_samples_split, min_samples_leaf in param_combinations:

            model_rf = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              random_state=1)
            model_rf.fit(X_train, Y_train)
            validation_score = model_rf.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [n_estimators, max_depth, min_samples_split, min_samples_leaf]


        # Test with the complete train data
        best_model = RandomForestClassifier(n_estimators=best_params[0],
                                          max_depth=best_params[1],
                                          min_samples_split=best_params[2],
                                          min_samples_leaf=best_params[3],
                                          random_state=1).fit(X_train_full, Y_train_full)
        best_model.fit(X_train_full, Y_train_full)


        test_score_rf = best_model.score(X_test, Y_test)
        print(f"Final Test Score: {test_score_rf}")

        # Save the results
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Test_score'] = test_score_rf

        params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
        df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Parameter'] = params_as_string

    ###################### SUPPORT VECTOR MACHINE ######################
    if support_vector_machine:
        model_rn = "svm"
        print(f"{ticker} is trained using SVM.")
        best_validation_score = 0

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

        for C, kernel, gamma in param_combinations_kernel:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma, random_state=1).fit(X_train, Y_train)
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
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
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [C, kernel, gamma, degree]



        # Test with the complete train data
        # if best the kernel "poly" was the best model:
        if len(best_params) == 4:
            best_model = svm.SVC(C=best_params[0], kernel=best_params[1], gamma=best_params[2], degree=best_params[3], random_state=1)
            best_model.fit(X_train_full, Y_train_full)

            test_score_svm = best_model.score(X_test, Y_test)
            print(f"Final Test Score: {test_score_svm}")

            # Save the results
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Test_score'] = test_score_svm

            params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Parameter'] = params_as_string

        # if kernel: linear, rbf, sigmoid
        else:
            best_model = svm.SVC(C=best_params[0], kernel=best_params[1], gamma=best_params[2], random_state=1)
            best_model.fit(X_train_full, Y_train_full)

            test_score_svm = best_model.score(X_test, Y_test)
            print(f"Final Test Score: {test_score_svm}")

            # Save the results
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Test_score'] = test_score_svm

            params_as_string = str([best_params[0], best_params[1], best_params[2]])
            df_hyperparameter.loc[(df_hyperparameter['Ticker'] == ticker) & (df_hyperparameter['Model'] == model_rn), 'Parameter'] = params_as_string

    if neural_network:
        pass
        # to be continued...



print(df_hyperparameter)

df_hyperparameter.to_csv(full_path + "data/funds_hyperparameter.csv", encoding="utf-8", index=True)