from src.load_data import SimpleDataset
from src.model import SimpleModel
from src.plot_history import plot_history
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os 
import pathlib
import logging
import time
import io
import threading

# Set the logging level and format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("training_log.log"),  # Logs will be saved to this file
        logging.StreamHandler()                  # Logs will also be printed to the console
    ]
)

# Load data

# Get the path of the training data
path = "data/train"
path = pathlib.Path(path)
file_paths_train = []
# iterate through the files in the path
for file in os.listdir(path):
    if file.endswith(".csv"):
        file_paths_train.append(os.path.join(path, file))

#-------------------------------------------------------------------------------------------------------#

# Get the path of the test data
path = "data/test"
path = pathlib.Path(path)
file_paths_test = []
# iterate through the files in the path
for file in os.listdir(path):
    if file.endswith(".csv"):
        file_paths_test.append(os.path.join(path, file))

#-------------------------------------------------------------------------------------------------------#

# Load the training data
dataset = SimpleDataset(file_paths=file_paths_train, lag=3, make_3d=True)
dataset.load_data()

# Scale the training data
dataset.scale_data()

# Create lagged features
dataset.create_lagged_features()

# Load the test data
dataset_test = SimpleDataset(file_paths=file_paths_test, lag=3, make_3d=True)
dataset_test.load_data()

# Scale the test data
dataset_test.scale_data()

# Create lagged features
dataset_test.create_lagged_features()

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = dataset.train_test_split_data()

# Prepare the test data
test_X, test_y = dataset_test.prepare_data()

print(f"Training feature shape: {train_X.shape}\nTraining target shape: {train_y.shape}")
print(f"Validation feature shape: {val_X.shape}\nValidation target shape: {val_y.shape}")
print(f"Test feature shape: {test_X.shape}\nTest target shape: {test_y.shape}")

# log the shapes of the data and first few rows of the data

logging.info(f"Training feature shape: {train_X.shape}\nTraining target shape: {train_y.shape}")
logging.info(f"Validation feature shape: {val_X.shape}\nValidation target shape: {val_y.shape}")
logging.info(f"Test feature shape: {test_X.shape}\nTest target shape: {test_y.shape}")

logging.info(f"Training features:\n{train_X[:5]}")
logging.info(f"Validation features:\n{val_X[:5]}")
logging.info(f"Training target:\n{train_y[:5]}")

#-------------------------------------------------------------------------------------------------------#
# Train the model


# Get the input shape
input_shape = (train_X[0].shape)

# Create and train the model

# Assign the default parameters for the model

params = {
    "input_shape": input_shape,
    "neurons": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "n_lstm_layers": 2,
    "batch_size": 128,
    "epochs": 200
}


# Function to get user input with a timeout
def input_with_timeout(prompt, timeout=10):
    def get_input():
        nonlocal user_input
        user_input = input(prompt)

    user_input = None
    thread = threading.Thread(target=get_input)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return 'n'
    return user_input

# Ask user if they want to change the default parameters
change_params = input_with_timeout("Do you want to change the default parameters? (Y/N): ")

if change_params.lower() == "y":
    params["neurons"] = int(input("Enter the number of neurons: "))
    params["learning_rate"] = float(input("Enter the learning rate: "))
    params["dropout_rate"] = float(input("Enter the dropout rate: "))
    params["n_lstm_layers"] = int(input("Enter the number of LSTM layers: "))
    params["batch_size"] = int(input("Enter the batch size: "))
    params["epochs"] = int(input("Enter the number of epochs: "))

# log the parameters of the model
logging.info(f"Model Parameters: {params}")

model = SimpleModel(**params)

# log the model summary

# Redirect the model summary to a string and log it
stream = io.StringIO()
model.model.summary(print_fn=lambda x: stream.write(x + "\n"))
summary_str = stream.getvalue()
logging.info(f"Model Summary:\n{summary_str}")
stream.close()


history = model.train(train_X, train_y, val_X, val_y)

# log the history of the model
logging.info(f"Model History : {history.history}")

# Plot the training history and save the plots
plot_history(history)

#-------------------------------------------------------------------------------------------------------#
# Evaluate the model


# Make predictions on the test data and evaluate the model
predictions = model.predict(test_X)

# Evaluate the model
eval_results = model.evaluate(test_X, test_y)
print("Evaluation Results:", eval_results)

# Inverse scaling of the predictions and actual values
predictions_inv = dataset.scaler_y.inverse_transform(predictions)

# Reshape test_y to 2D before inverse scaling
y_test_reshaped = test_y.values.reshape(-1, 1)
y_test_inv = dataset.scaler_y.inverse_transform(y_test_reshaped)

# Calculate the evaluation metrics
mse = mean_squared_error(y_test_inv, predictions_inv)
mae = mean_absolute_error(y_test_inv, predictions_inv)

# Log the evaluation metrics
logging.info(f"MSE: {mse}")
logging.info(f"MAE: {mae}")
logging.info(f"RMSE: {np.sqrt(mse)}")

# Print the evaluation metrics
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", np.sqrt(mse))

#---------------------------------------------------------------------------------------------#
# Save the model if the evaluation metrics less than 400 RMSE

if np.sqrt(mse) < 400:
    model.save_model(f"model/Gas_turbine_EP{np.sqrt(mse)}.keras")
    logging.info(f"Model saved successfully! to model/Gas_turbine_EP{np.sqrt(mse)}.keras with {np.sqrt(mse)} RMSE")

else:
    logging.info(f"Model not saved! RMSE is greater than 400 | RMSE: {np.sqrt(mse)}")
    print("Model not saved! RMSE is greater than 400")