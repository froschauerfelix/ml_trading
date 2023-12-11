# generate all the data for the models

from config import start, end, full_path
import yfinance as yf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# get tickers
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
ticker_list = list(tickers.Symbol)[0:20] # small subset [0:20]

# get data from tickers
data_first = yf.Ticker("AMZN")
df = data_first.history(start=start, end=end)
df["ticker"] = "AMZN"

for ticker in ticker_list:
    if ticker != "AMZN":
        # print(ticker)
        data = yf.Ticker(ticker)
        stock_data = data.history(start=start, end=end)
        stock_data["ticker"] = ticker
        frames = [df, stock_data]
        df = pd.concat(frames)

# label the data set
df["label"] = np.where(df.index > df.index[756], "test", "train")

# save the file
stock_data_path = full_path + "data/stock_data.csv"
df.to_csv(stock_data_path)

# plot example
subset = df[df.ticker == "AMZN"]

fig, ax = plt.subplots()

groups = subset.groupby("label")
for name, group in groups:
    ax.plot(group.index, group.Close, label=name)

ax.axvline(x=subset.index[756], linewidth=2, color='k')

plt.title("Amazon stock history")
plt.legend()
plt.xticks(rotation=90)
plt.show()
