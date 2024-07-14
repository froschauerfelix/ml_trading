# ensure that the file can be executed in console
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ml_trading.code.config import full_path, start, end, tickers
import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import math
from rich import print

# Normal:
# SPY = S&P 500
# QQQ = Nasdaq

# SMALL:
# iShares Russell 2000 ETF (IWM)
# Vanguard Small-Cap ETF (VB)

# Mikro:
# iShares Micro-Cap ETF (IWC)
# Invesco S&P SmallCap 600 Pure Value ETF (RZV)


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
print(df.head(n=5))


print('[green]The data generation process is finished. You can now continue with the preprocessing file.[/]')
