# A list for most variables
#

# Define your full path to the project folder
full_path = "/Users/felix/wichtiges/AAA Master/Studium/WiSe 23%24/Master Thesis/programming/ml_trading/"


# Time frame for the required data
start = "2012-01-01"
end = "2022-12-31"

# Fund ticker symbols
tickers = ["SPY", "QQQ", "IWM", "VB", "IWC", "FDM"]

"""
Market ETF:
SPY = S&P 500 
QQQ = Nasdaq

Small Cap ETF:
IWM = Russell 2000 ETF
VB = Vanguard Small Cap ETF

Micro Cap ETF:
IWC = iShares Micro-Cap ETF
FDM = Ivesco S&P SmallCap 600 Pure Value ETF
"""


# Time frame for validation and testing
train_end = "2019-01-01"
validate_end = "2020-01-01"

"""
Training: from 2012-01-01 until 2018-12-31
Validation: from 2019-01-01 until 2019-12-31
Testing: from 2020-01-01 until 2022-12-31
"""

# Features created from the data and used as input in the models
features = ["SMA_15_minus_SMA_50", "MACD_Line", "MACD_Histogram", "RSI", "OBV_MA_diff", "SMA_15",
            "SMA_50", "EMA_12", "EMA_26", "MACD_Signal", "OBV", "OBVMA"]

# Selects which models are trained given the data
random_forest = True
support_vector_machine = True
neural_network = True

# don't change anything here
models = [random_forest, support_vector_machine, neural_network]


# Number of iterations in the neural network
number_epochs = 100

# Trading costs
with_costs = True

# Cost structure per ETF (in %)
average_costs_etf = {
    "SPY": 0.003,
    "QQQ": 0.004,
    "IWM": 0.005,
    "VB": 0.026,
    "IWC": 0.31,
    "FDM": 0.43,
}


