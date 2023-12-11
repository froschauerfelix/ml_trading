# preprocess and output data again

import pandas as pd
from config import full_path

from matplotlib import pyplot as plt

import_path = full_path + "data/stock_data.csv"
df = pd.read_csv(import_path)

# first ticker
ticker = "AMZN"
small_df = df[df.ticker == ticker]

small_df = small_df.assign(close_diff_1=small_df['Close'].shift(1) - small_df['Close'].shift(2))
small_df = small_df.assign(close_diff_2=small_df['Close'].shift(2) - small_df['Close'].shift(3))
small_df = small_df.assign(volume_diff_1=small_df['Volume'].shift(1) - small_df['Volume'].shift(2))
small_df = small_df.assign(volume_diff_2=small_df['Volume'].shift(2) - small_df['Volume'].shift(3))

small_df = small_df.assign(close_shifted=small_df["Close"].shift(1))  # shift by one, such that the MA doesn't include t
small_df = small_df.assign(MA_5=small_df["close_shifted"].rolling(5).mean())
small_df = small_df.assign(MA_10=small_df["close_shifted"].rolling(10).mean())
small_df = small_df.assign(MA_30=small_df["close_shifted"].rolling(30).mean())

# for the remaining tickers

for ticker in list(df.ticker.unique()):
    if ticker != "AMZN":
        new_df = df[df.ticker == ticker]

        new_df = new_df.assign(close_diff_1=new_df['Close'].shift(1) - new_df['Close'].shift(2))
        new_df = new_df.assign(close_diff_2=new_df['Close'].shift(2) - new_df['Close'].shift(3))
        new_df = new_df.assign(volume_diff_1=new_df['Volume'].shift(1) - new_df['Volume'].shift(2))
        new_df = new_df.assign(volume_diff_2=new_df['Volume'].shift(2) - new_df['Volume'].shift(3))

        new_df = new_df.assign(close_shifted=new_df["Close"].shift(1))
        new_df = new_df.assign(MA_5=new_df["close_shifted"].rolling(5).mean())
        new_df = new_df.assign(MA_10=new_df["close_shifted"].rolling(10).mean())
        new_df = new_df.assign(MA_30=new_df["close_shifted"].rolling(30).mean())

        frames = [small_df, new_df]
        small_df = pd.concat(frames)

print(f"Number of rows: {len(small_df)}")

# plotting
subset = small_df[small_df.ticker == "AMZN"]

fig, ax = plt.subplots()

groups = subset.groupby("label")
for name, group in groups:
    ax.plot(group.index, group.Close, label=name)

ax.plot(subset.index, subset.MA_30, c="darkgreen", label="MA_30")
ax.axvline(x=subset.index[756], linewidth=2, color='k')

plt.title("Amazon stock history")
plt.legend()
plt.xticks(rotation=90)
plt.show()


# save the file
stock_data_path = full_path + "data/stock_data_preprocessed.csv"
small_df.to_csv(stock_data_path)

print(small_df.head())