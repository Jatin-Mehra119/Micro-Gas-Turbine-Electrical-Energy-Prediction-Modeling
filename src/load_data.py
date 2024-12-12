import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class SimpleDataset:
    def __init__(self, file_paths, lag=1, make_3d=False):
        self.file_paths = file_paths  # List of file paths
        self.lag = lag
        self.make_3d = make_3d
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.data = None

    def load_data(self):
        """Load the data from the list of files."""
        data_frames = []
        for file_path in self.file_paths:
            data = pd.read_csv(file_path)
            data_frames.append(data)
        self.data = pd.concat(data_frames, ignore_index=True)

    def scale_data(self):
        """Scale the input and output data."""
        self.data_scaled = self.data.copy()
        self.data_scaled[['input_voltage']] = self.scaler_x.fit_transform(self.data[['input_voltage']])
        self.data_scaled[['el_power']] = self.scaler_y.fit_transform(self.data[['el_power']])

    def create_lagged_features(self):
        """Create lagged features based on the specified lag."""
        for i in range(1, self.lag + 1):
            self.data_scaled[f'input_voltage(t-{i})'] = self.data_scaled['input_voltage'].shift(i)
        self.data_scaled.dropna(inplace=True)

    def prepare_data(self):
        """Prepare the final data for training."""
        # Create the input and output data
        X = self.data_scaled.drop(columns=['el_power'])
        y = self.data_scaled['el_power']

        # Reshape to 3D if needed for deep learning
        if self.make_3d:
            X = X.values.reshape((X.shape[0], X.shape[1], 1))

        return X, y

    def train_test_split_data(self):
        """Split the data into training and testing sets."""
        X, y = self.prepare_data()
        return train_test_split(X, y, test_size=0.2, random_state=42)