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

tickers = list(funds_info.fund_ticker)
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


# visualize the tickers (optional)
if plotting == "yes":
    fig, axs = plt.subplots(math.ceil(len(tickers) / 2), 2)

    for idx, ticker in enumerate(tickers):
        sub_df = df[df["Ticker"] == ticker]

        row = idx % 3
        col = idx % 2

        axs[row, col].plot(sub_df.index.values, sub_df.Open.values)
        axs[row, col].set_title(f"Ticker: {ticker}")
        for ax in axs.flat:
            ax.label_outer()

    plt.tight_layout()
    plt.show()


