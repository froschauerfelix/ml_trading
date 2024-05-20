# List of all variables that can be changed in this file.


# Define your full path to the project folder
full_path = "/Users/felix/wichtiges/AAA Master/Studium/WiSe 23%24/Master Thesis/programming/ml_trading/"

# Time Period (train+validate+test)
start = "2012-01-01"
end = "2022-12-31" # 2019

# validate_end > train_end!
train_end = "2019-01-01"
validate_end = "2020-01-01"



# Selects which model is trained in model_tuning.py
random_forest = True
support_vector_machine = True
neural_network = True

# don't change anything here
models = [random_forest, support_vector_machine, neural_network]

# select all features the models are trained with


#features = ["DP_1", "DP_2", "DV_1", "DV_2", "SMA_10", "SMA_25", "SMA_50", "EMA_12", "EMA_26", "MACD_Line", "MACD_Signal", "MACD_Histogram", "RSI", "OBV", "OBVMA"]
features = ["SMA_10", "SMA_25", "SMA_50", "EMA_12", "EMA_26", "MACD_Line", "MACD_Signal", "MACD_Histogram", "RSI", "OBV", "OBVMA"]



# DP_1, DP_2, DV_1, DV_2, MA_5, MA_10, MA_30, MPP_30, MPP_50 => from Chiang et al 2016
# RSI, %K, %D => Thomsett 2018

# Neural Network
number_epochs = 100

# Calculate Return with trading costs?
with_costs = True
