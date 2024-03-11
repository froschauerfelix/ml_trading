# ensure that the file can be executed in console
import os, sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path, train_end, models

import pandas as pd
from rich import print
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import itertools

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import tree


# Select type of model to train
random_forest = models[0]
support_vector_machine = models[1]
neural_network = models[2]



# Import data
funds_processed = pd.read_csv(full_path + "data/funds_processed.csv", index_col=0)
funds_return = pd.read_csv(full_path + "data/funds_return.csv", index_col=0)
tickers = list(funds_return.Ticker.unique())


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
df_result = pd.DataFrame(zip(ticker_column, model_column), columns=["Ticker", "Model"])
df_result["Test_score"] = np.nan
df_result["Parameter"] = np.nan




for ticker in tickers:

    ticker_return = funds_processed[funds_processed["Ticker"] == ticker]

    ticker_train = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index < train_end)]
    ticker_validate = ticker_return[(ticker_return["Type"] == "train") & (ticker_return.index > train_end)]
    ticker_test = ticker_return[ticker_return["Type"] == "test"]

    X_train = ticker_train.drop(["Ticker", "Target_tomorrow", "Type"], axis=1)
    Y_train = ticker_train["Target_tomorrow"]

    X_validate = ticker_validate.drop(["Ticker", "Target_tomorrow", "Type"], axis=1)
    Y_validate = ticker_validate["Target_tomorrow"]

    X_test = ticker_test.drop(["Ticker", "Target_tomorrow", "Type"], axis=1)
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
            #print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

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


        # test the results with the test set for the first time
        best_model = RandomForestClassifier(n_estimators=best_params[0],
                                          max_depth=best_params[1],
                                          min_samples_split=best_params[2],
                                          min_samples_leaf=best_params[3],
                                          random_state=1)
        best_model.fit(X_train, Y_train)


        test_score_rf = best_model.score(X_test, Y_test)
        print(f"Final Test Score: {test_score_rf}")

        # Save the results
        df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Test_score'] = test_score_rf

        params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
        df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Parameter'] = params_as_string

    ###################### SUPPORT VECTOR MACHINE ######################
    if support_vector_machine:
        model_rn = "svm"
        print(f"{ticker} is trained using SVM.")
        best_validation_score = 0


        # One Grid for each Kernel
        # LINEAR KERNEL
        param_grid_linear = {
            'C': [0.1, 1, 10], # 100
            'kernel': ['linear'],
            'gamma': ['scale', 'auto'],
        }
        param_combinations_linear = itertools.product(
            param_grid_linear['C'],
            param_grid_linear['kernel'],
            param_grid_linear['gamma']
        )

        for C, kernel, gamma in param_combinations_linear:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma).fit(X_train, Y_train)
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [C, kernel, gamma]



        # RBF KERNEL
        param_grid_rbf = {
            'C': [0.1, 1, 10], # 100
            'kernel': ['rbf'],
            'gamma': [0.001, 0.0001, 'scale', 'auto'],
        }
        param_combinations_rbf = itertools.product(
            param_grid_rbf['C'],
            param_grid_rbf['kernel'],
            param_grid_rbf['gamma']
        )

        for C, kernel, gamma in param_combinations_rbf:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma).fit(X_train, Y_train)
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [C, kernel, gamma]


        # POLY KERNEL
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
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma, degree=degree).fit(X_train, Y_train)
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [C, kernel, gamma, degree]


        # SIGMOID KERNEL
        param_grid_sigmoid = {
            'C': [0.1, 1, 10], # 100
            'kernel': ['sigmoid'],
            'gamma': [0.001, 0.0001, 'scale', 'auto'],
        }
        param_combinations_sigmoid = itertools.product(
            param_grid_sigmoid['C'],
            param_grid_sigmoid['kernel'],
            param_grid_sigmoid['gamma']
        )

        for C, kernel, gamma in param_combinations_sigmoid:
            model_svm = svm.SVC(C=C, kernel=kernel, gamma=gamma).fit(X_train, Y_train)
            validation_score = model_svm.score(X_validate, Y_validate)

            # Save the parameters of the best validation score
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_params = [C, kernel, gamma]



        # test the results with the test set for the first time
        if len(best_params) == 4:
            best_model = svm.SVC(C=best_params[0], kernel=best_params[1], gamma=best_params[2], degree=best_params[3]).fit(X_train, Y_train)
            test_score_svm = best_model.score(X_test, Y_test)
            print(f"Final Test Score: {test_score_svm}")

            # Save the results
            df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Test_score'] = test_score_svm

            params_as_string = str([best_params[0], best_params[1], best_params[2], best_params[3]])
            df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Parameter'] = params_as_string

        else:
            best_model = svm.SVC(C=best_params[0], kernel=best_params[1], gamma=best_params[2]).fit(X_train, Y_train)
            test_score_svm = best_model.score(X_test, Y_test)
            print(f"Final Test Score: {test_score_svm}")

            # Save the results
            df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Test_score'] = test_score_svm

            params_as_string = str([best_params[0], best_params[1], best_params[2]])
            df_result.loc[(df_result['Ticker'] == ticker) & (df_result['Model'] == model_rn), 'Parameter'] = params_as_string




    if neural_network:
        print("hi")


    print(df_result)


df_result.to_csv(full_path + "data/funds_results.csv", encoding="utf-8", index=True)



"""
        model_rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, verbose=1, cv=2)
        grid_search.fit(X_train, Y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        test_score = best_model.score(X_test, Y_test)
        train_score = best_model.score(X_train, Y_train)



# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(best_params)

test_score = best_model.score(X_test, Y_test)
train_score = best_model.score(X_train, Y_train)

print(f"Test Accuracy: {test_score}")
print(f"Train Accuracy: {train_score}")


pred = best_model.predict(X_test)

pred_proba = best_model.predict_proba(X_test)[:, 1]
#predict_labels = ['SELL' if x >= 0.6 else 'HOLD' if x >= 0.4 else 'BUY' for x in pred_proba]


precision = precision_score(Y_test, pred)

print(f"Test Precision: {precision}")


print("Value_counts_pred:")
print(pd.Series(pred).value_counts())
print("Value_counts_real:")
print(Y_test.value_counts())


print("first max, then min")
print(pred_proba.max())
print(pred_proba.min())


funds_test["Signal"] = pred
funds_test["Daily_Return"] = (funds_test["Close"] - funds_test["Open"])/funds_test["Open"] # %daily return

funds_test.to_csv(full_path + "data/funds_predictions.csv", encoding="utf-8", index=True)
print('[green]The predictions are made. To find the actual returns, go to the back-testing file.[/]')

"""

# Visualize confusion matrix
"""cm = confusion_matrix(Y_test, pred)

# Plot confusion matrix
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()"""


"""
# visualize tree
single_tree = best_model.estimators_[0]
fn = X_train.columns
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=400)

# Now plot the single tree
tree.plot_tree(single_tree,
               feature_names=fn,
               filled=True)

plt.tight_layout()
plt.show()"""

"""
param_grid = {
    'n_estimators': [100, 200, 500], # Number of trees in the forest
    'max_depth': [None, 10, 20, 30], # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required at each leaf node
    'bootstrap': [True, False] # Method of selecting samples for training each tree
}
"""

"""

# for the back-testing
funds_return = funds_return[funds_return["Ticker"] == "SPY"]
funds_test = funds_return[funds_return.index >= train_end]

# drop last row such that the length match
funds_test = funds_test.drop(funds_test.index[-1])

"""