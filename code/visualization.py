import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from ml_trading.code.config import full_path

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
import tikzplotlib
np.random.seed(41)



df_predictions = pd.read_csv(full_path + "data/funds_predictions.csv", index_col=0)

# only one at a time
support_vector_visualization = False
time_series_data = False
trading_activity = True


if support_vector_visualization:
    #plt.switch_backend('pgf')

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
    #plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot data points
    ax.scatter(X['x_1'][y == 0], X['x_2'][y == 0], color='red', label='Class 0', marker="x")
    ax.scatter(X['x_1'][y == 1], X['x_2'][y == 1], color='blue', label='Class 1', marker="x")

    # Plot the separating hyperplane
    ax.plot(xx, yy, 'k-', label='Separating hyperplane')

    # Plot margin lines
    ax.plot(xx, yy_down, 'k--', label='Margin')
    ax.plot(xx, yy_up, 'k--')

    # Highlight the support vectors
    #plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
             #   s=100, facecolors='none', edgecolors='k', label='Support vectors')

    ax.set_xlabel("Input Feature 1")
    ax.set_ylabel("Input Feature 1")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.legend()
    ax.grid(True)


    tikzplotlib.save(full_path + "output/svm_visualization.tex")
    plt.show()


if time_series_data:
    # Load the data from CSV
    funds_return = pd.read_csv(full_path + "data/funds_data_raw.csv", index_col=0)
    funds_return.index = pd.to_datetime(funds_return.index).copy()

    # Extract unique tickers
    tickers = list(funds_return.Ticker.unique())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot data for each ticker
    for ticker in tickers:
        ticker_data = funds_return[funds_return['Ticker'] == ticker]
        ax.plot(ticker_data.index, ticker_data['Close'], label=ticker)

    # Define the list of years for x-tick labels
    years = pd.date_range(start='2012-01-01', end='2022-01-01', freq='YS').year

    # Set the x-ticks to the list of years and format them to display only the year
    ax.set_xticks(pd.to_datetime(years, format='%Y'))
    ax.set_xticklabels(years, rotation=45)

    # Add labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    # Save the plot
    tikzplotlib.save(full_path + "output/tickers_visualization.tex")
    plt.show()


if trading_activity:
    subs = df_predictions[(df_predictions.Ticker == "IWC") & (df_predictions.Model == "random_forest")]
    print(subs.head())

    subs.index = pd.to_datetime(subs.Date).copy()
    plt.figure(figsize=(14, 7))
    plt.plot(subs.index, subs['Open'], label='Close Price', color='skyblue')
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

    plt.title('Stock Market Predictions with Buy Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()




