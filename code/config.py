# set variables


# Define your full path to the project folder
full_path = "/Users/felix/wichtiges/AAA Master/Studium/WiSe 23%24/Master Thesis/programming/ml_trading/"

# Time Period
start = "2008-01-01"
end = "2019-12-31"

# validate_end > train_end
train_end = "2016-01-01"
validate_end = "2018-01-01"


# Selects which model is trained in model_hyperparameter.py
random_forest = True
support_vector_machine = True
neural_network = False

# don't change anything here
models = [random_forest, support_vector_machine, neural_network]

# select features
features = ["DP_1", "DP_2", "DV_1", "DV_2", "MA_5", "MA_10", "MA_30", "MPP_30", "MPP_50"]
