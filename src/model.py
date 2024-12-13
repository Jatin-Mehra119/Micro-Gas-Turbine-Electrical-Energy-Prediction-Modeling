import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
import pandas as pd
from datetime import datetime

class SimpleModel:
    """
    A class to create and train an LSTM model for electrical energy prediction.

    Attributes:
    ----------
    neurons : int
        Number of neurons in each LSTM layer.
    learning_rate : float
        Learning rate for the optimizer.
    dropout_rate : float
        Dropout rate for regularization.
    n_lstm_layers : int
        Number of LSTM layers in the model.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of epochs for training.
    model : tf.keras.Sequential
        The LSTM model.

    Methods:
    -------
    build_model(input_shape):
        Builds the LSTM model based on the provided input shape.
    compile_model():
        Compiles the model with the specified optimizer and loss function.
    train_model(x_train, y_train, x_val, y_val):
        Trains the model on the training data and validates on the validation data.
    """
    def __init__(self, input_shape, neurons=32, learning_rate=0.001, dropout_rate=0.2, n_lstm_layers=2, batch_size=128, epochs=100):
        """
        Constructs all the necessary attributes for the SimpleModel object.

        Parameters:
        ----------
        input_shape : tuple
            Shape of the input data.
        neurons : int, optional
            Number of neurons in each LSTM layer (default is 32).
        learning_rate : float, optional
            Learning rate for the optimizer (default is 0.001).
        dropout_rate : float, optional
            Dropout rate for regularization (default is 0.2).
        n_lstm_layers : int, optional
            Number of LSTM layers in the model (default is 2).
        batch_size : int, optional
            Batch size for training (default is 128).
        epochs : int, optional
            Number of epochs for training (default is 100).
        """
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.n_lstm_layers = n_lstm_layers
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Model Creation
        model = tf.keras.Sequential()
        if self.n_lstm_layers == 1:
            model.add(layers.LSTM(self.neurons, input_shape=input_shape, return_sequences=False))
        else:
            model.add(layers.LSTM(self.neurons, input_shape=input_shape, return_sequences=True))
            model.add(layers.Dropout(self.dropout_rate))
            for i in range(self.n_lstm_layers - 2):
                model.add(layers.LSTM(self.neurons, return_sequences=True))
                model.add(layers.Dropout(self.dropout_rate))
            model.add(layers.LSTM(self.neurons, return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(self.neurons // 2, activation="relu"))
        model.add(layers.Dense(1, activation="linear"))
        
        # Compile Model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
            
        self.model = model

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Trains the model on the training data and validates on the validation data.

        Parameters:
        ----------
        x_train : array-like
            Training input data.
        y_train : array-like
            Training target data.
        x_val : array-like, optional
            Validation input data (default is None).
        y_val : array-like, optional
            Validation target data (default is None).

        Returns:
        -------
        history : History
            A record of training loss values and metrics values at successive epochs.
        """
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                                 epochs=self.epochs, batch_size=self.batch_size, 
                                 callbacks=[early_stop], verbose=1)
        return history

    def predict(self, x_test):
        """
        Predicts the target values for the test data.

        Parameters:
        ----------
        x_test : array-like
            Test input data.
        
        Returns:
        -------
        predictions : array-like
            Predicted target values.
        """
        predictions = self.model.predict(x_test)
        return predictions

    def save_model(self, filepath):
        """
        Saves the model to the specified filepath.

        Parameters:
        ----------
        filepath : str
            Path to save the model.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Loads the model from the specified filepath.

        Parameters:
        ----------
        filepath : str
            Path to load the model from.
        """
        self.model = tf.keras.models.load_model(filepath)

    def evaluate(self, x_test, y_test):
        """
        Evaluates the model on the test data.

        Parameters:
        ----------
        x_test : array-like
            Test input data.
        y_test : array-like
            Test target data.

        Returns:
        -------
        eval_results : list
            Evaluation results.
        """
        eval_results = self.model.evaluate(x_test, y_test, verbose=1)
        return eval_results
    
    def summary(self):
        """
        Returns a summary of the model architecture. (For logging purposes)
        """
        self.model.summary()