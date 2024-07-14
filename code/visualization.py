import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import tikzplotlib
from rich import print

# just relevant for svm visualization
np.random.seed(41)


df_funds = pd.read_csv(full_path + "data/funds_data_raw.csv", index_col=0)
df_funds_processed = pd.read_csv(full_path + "data/funds_data_processed.csv", index_col=0)
df_predictions = pd.read_csv(full_path + "data/funds_predictions_with_costs.csv", index_col=0)


# Select which Visualization should be produced and outputted
support_vector_visualization = False
time_series_data = False
time_series_results = False
trading_activity = True

# "SPY", "QQQ", "IWM", "VB", "IWC", "FDM"
trading_ticker = "FDM"

#"random_forest", "svm", "neural_network"
trading_model = "random_forest"


if support_vector_visualization:
    # Generate random data
    num_points = 10
    x_1 = np.random.normal(1, 1, num_points)
    x_2 = np.random.normal(1, 2, num_points)
    x_3 = np.random.normal(2.5, 0.5, num_points)
    x_4 = np.random.normal(2, 1, num_points)

    data = pd.concat([
        pd.DataFrame({'x_1': x_1, 'x_2': x_2, 'y': 0}),
        pd.DataFrame({'x_1': x_3, 'x_2': x_4, 'y': 1})
    ])

    X = data[['x_1', 'x_2']]
    y = data['y']

    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # Get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(X['x_1'].min(), X['x_1'].max())
    yy = a * xx - (clf.intercept_[0]) / w[1]

    # Calculate margin lines
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + np.sqrt(1 + a ** 2) * margin
    yy_up = yy - np.sqrt(1 + a ** 2) * margin

    # Plot the data points and the separating hyperplane
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data points
    ax.scatter(X['x_1'][y == 0], X['x_2'][y == 0], color='red', label='Class 0', marker="x")
    ax.scatter(X['x_1'][y == 1], X['x_2'][y == 1], color='blue', label='Class 1', marker="x")

    # Plot the separating hyperplane
    ax.plot(xx, yy, 'k-', label='Separating hyperplane')

    # Plot margin lines
    ax.plot(xx, yy_down, 'k--', label='Margin')
    ax.plot(xx, yy_up, 'k--')


    ax.set_xlabel("Input Feature 1")
    ax.set_ylabel("Input Feature 1")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.legend()
    ax.grid(True)

    # save as tex file (can be inputted in latex)
    tikzplotlib.save(full_path + "output/svm_visualization.tex")
    plt.show()


if time_series_data:

    # Change index to datetime
    df_funds.index = pd.to_datetime(df_funds.index).copy()

    # Extract unique tickers
    tickers = list(df_funds.Ticker.unique())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot data for each ticker
    for ticker in tickers:
        ticker_data = df_funds[df_funds['Ticker'] == ticker]
        ax.plot(ticker_data.index, ticker_data['Close'], label=ticker)

    # Define the list of years for x-tick labels
    years = pd.date_range(start='2012-01-01', end='2023-01-01', freq='YS').year

    # Set the x-ticks to the list of years and format them to display only the year
    ax.set_xticks(pd.to_datetime(years, format='%Y'))
    ax.set_xticklabels(years, rotation=0)

    # Add labels and legend
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    # Save the plot as tex file
    tikzplotlib.save(full_path + "output/tickers_visualization.tex")

    plt.show()


if time_series_results:

    tickers = ["SPY", "QQQ", "IWM", "VB", "IWC", "FDM"]

    for ticker in tickers:

        holding_data = df_funds_processed
        holding_data.index = pd.to_datetime(holding_data.index).copy()

        funds_return = df_predictions
        funds_return.index = pd.to_datetime(funds_return.Date).copy()


        # Select the right period
        holding_data = holding_data[holding_data.Type == "test"]

        # Subset
        holding_data = holding_data[holding_data.Ticker == ticker]
        funds_return = funds_return[funds_return["Ticker"] == ticker]

        # Transform the to return in the buy and hold
        initial_price = holding_data["Close"].iloc[0]
        holding_data["buy_hold_return"] = holding_data["Close"] / initial_price

        # Extract unique tickers
        models = list(funds_return.Model.unique())

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 3))



        colors = ["tab:orange", "tab:green", "tab:red"]
        x = 0

        ax.plot(holding_data.index, holding_data["buy_hold_return"], label="Buy&Hold", color="tab:blue")

        # Plot data for each ticker
        for model in models:

            ticker_data = funds_return[funds_return['Model'] == model]
            ax.plot(ticker_data.index, ticker_data['current_return'], label=model, color=colors[x])
            #ax.plot(ticker_data.index, ticker_data["Close"], label=ticker)
            x += 1
        years = pd.date_range(start='2020-01-01', end='2023-01-01', freq='YS').year

        # Set the x-ticks to the list of years and format them to display only the year
        ax.set_xticks(pd.to_datetime(years, format='%Y'))
        ax.set_xticklabels(years, rotation=0)

        # Add labels and legend
        plt.title(ticker)
        ax.set_xlabel("Year")
        ax.set_ylabel("Return")
        ax.legend()#loc="upper left")
        ax.grid(True)

        # Save the plot
        name = "output/ticker_" + ticker + "_return.tex"

        tikzplotlib.save(full_path + name)

        plt.show()


if trading_activity:
    pred_ticker = trading_ticker
    pred_model = trading_model

    # Create subset
    subs = df_predictions[(df_predictions.Ticker == pred_ticker) & (df_predictions.Model == pred_model)]
    subs.index = pd.to_datetime(subs.Date).copy()

    plt.figure(figsize=(10, 3))
    years = [2020, 2021, 2022, 2023]

    plt.plot(subs.index, subs['Open'], label='Opening Price', color='skyblue')
    plt.scatter(subs[subs['Adjusted_Signal'] == "SELL"].index, subs[subs['Adjusted_Signal'] == "SELL"]['Open'],
                label='Sell Signal', color='red', marker='^')
    plt.scatter(subs[subs['Adjusted_Signal'] == "BUY"].index, subs[subs['Adjusted_Signal'] == "BUY"]['Open'],
                label='Buy Signal', color='green', marker='^')

    # Initialize holding state and start_date
    holding = False
    start_date = subs.index[0]
    # Add shading for periods where stocks are not held
    for i in range(len(subs)):
        if subs['Adjusted_Signal'].iloc[i] == 'BUY' and not holding:
            end_date = subs.index[i]
            # Shade the period of not holding with red color
            plt.axvspan(start_date, end_date, color='red', alpha=0.1)
            start_date = subs.index[i]
            holding = True
        elif subs['Adjusted_Signal'].iloc[i] == 'SELL' and holding:
            end_date = subs.index[i]
            # Shade the period of holding with green color
            plt.axvspan(start_date, end_date, color='green', alpha=0.1)
            start_date = subs.index[i]
            holding = False

    # If the last action was a 'BUY', shade until the end of the dataset
    if holding:
        plt.axvspan(start_date, subs.index[-1], color='green', alpha=0.1)
    else:
        plt.axvspan(start_date, subs.index[-1], color='red', alpha=0.1)

    #plt.title(f'Stock Market Predictions with Buy Signals, {pred_ticker, pred_model}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    #plt.xticks(years, rotation=45)

    plt.xticks(rotation=0)
    plt.xticks(pd.to_datetime(years, format='%Y').to_pydatetime(), years)
    plt.tight_layout()


    file_name = full_path + "output/trading_" + trading_ticker + "_" + trading_model + ".tex"
    print(file_name)

    # Safe the plot as tex file
    tikzplotlib.save(file_name)

    plt.show()







