import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
import pandas as pd
from datetime import datetime

class SimpleModel:
    def __init__(self, input_shape, neurons=32, learning_rate=0.001, dropout_rate=0.2, n_lstm_layers=2, batch_size=128, epochs=100):
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
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                                 epochs=self.epochs, batch_size=self.batch_size, 
                                 callbacks=[early_stop], verbose=1)
        return history

    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

    def evaluate(self, x_test, y_test):
        eval_results = self.model.evaluate(x_test, y_test, verbose=1)
        return eval_results
    
    def summary(self):
        self.model.summary()