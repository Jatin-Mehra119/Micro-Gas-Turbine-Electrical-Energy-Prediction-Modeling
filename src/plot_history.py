import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    """
    Plots the training and validation loss and mean absolute error (MAE) over epochs.

    Parameters:
    ----------
    history : History
        A History object returned by the fit method of a Keras model. It contains the training and validation loss and MAE values over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.figure(figsize=(10, 5))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/loss_plot.png')
    plt.show()

    # Plot training & validation mae values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model mae')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/mae_plot.png')
    plt.show()

