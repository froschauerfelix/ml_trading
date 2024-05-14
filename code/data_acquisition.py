# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path, start, end
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import math
from rich import print

# import already existing dataset


funds_info = pd.read_csv(full_path + "data/funds_info.csv")

#tickers = list(funds_info.fund_ticker)

#tickers = ["SPY", "VOO", "MDY", "IWM", "RMIC"]
    
tickers = ["SPY", "QQQ", "IWC"]

# SPY = S&P 500
# QQQ = Nasdaq


# MDY = S&P  midcap
# IWM = russel index
# Russell Microcap = RUMIC
# ishares micro cap etf


dfs = []

# download data from yahoo finance and create a dataset for each ticker
for ticker in tickers:
    ticker_df = yf.download(ticker, start=start, end=end)
    ticker_df["Ticker"] = ticker
    dfs.append(ticker_df)

# combine all ticker_df to one big dataset
df = pd.concat(dfs)

# save the data
df.to_csv(full_path + "data/funds_data_raw.csv", encoding="utf-8", index=True)


print('[green]The data generation process is finished. You can now continue with the preprocessing file.[/]')
print(df)
plotting = input("Do you want to see the historical plot of the data you just downloaded? (yes/no)")

# Adjusted visualization part
if plotting == "yes":
    # Calculate the number of rows and columns for the subplot
    num_plots = len(tickers)
    num_columns = 2
    num_rows = math.ceil(num_plots / num_columns)

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 5))  # Adjust figsize as needed

    for idx, ticker in enumerate(tickers):
        sub_df = df[df["Ticker"] == ticker]

        row = idx // num_columns  # Corrected row calculation
        col = idx % num_columns  # Column calculation remains the same

        axs[row, col].plot(sub_df.index, sub_df['Open'])
        axs[row, col].set_title(f"Ticker: {ticker}")

    # This part ensures that any extra subplots not used are turned off
    for idx in range(num_plots, num_rows * num_columns):
        axs.flat[idx].axis('off')

    plt.tight_layout()
    plt.show()