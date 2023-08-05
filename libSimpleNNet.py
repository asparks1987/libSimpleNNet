import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
import numpy as np

def preprocess_data(df, target_col):
    """
    Preprocess data, converting string inputs to numpy arrays and encoding categorical target labels.

    Args:
        df: DataFrame containing the input and target data.
        target_col: Column name of the target data in the dataframe.

    Returns:
        X: Numpy array of input data.
        y: Numpy array of target data.
    """
    # Check if input data is already numpy array, if not convert string to numpy array
    if not isinstance(df['input'][0], np.ndarray):
        df['input'] = df['input'].apply(lambda x: np.array([float(i) for i in x.split(",")]))
    
    # Label encode the target column
    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col])

    # Convert input and target data to numpy arrays
    X = np.stack(df.drop(target_col, axis=1)['input'].values)
    y = df[target_col].values
    return X, y

def preprocess_data_array(input_array):
    """
    Convert input array to numpy array if it is not.

    Args:
        input_array: Input data array.

    Returns:
        numpy array of input data.
    """
    # If input data is not numpy array, convert it
    if not isinstance(input_array, np.ndarray):
        input_array = np.array([float(x) for x in input_array.split(",")])
    return input_array

def create_model(input_shape):
    """
    Creates LSTM model with given input shape.

    Args:
        input_shape: Shape of the input data.

    Returns:
        Created LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(X, y, callback=None, epochs=1):
    """
    Trains the LSTM model with given training data.

    Args:
        X: Input data.
        y: Target data.
        callback: Keras callback for updating progress.
        epochs: Number of training epochs.

    Returns:
        Trained model and history of the training.
    """
    # Check if input data is 1D, if so reshape it to 2D as LSTM expects input data to be 3D (batch_size, timesteps, features)
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # LSTM expects input data to be 3D (batch_size, timesteps, features)
    # If input data is 2D, reshape it to 3D
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    if len(X_test.shape) == 2:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = create_model((X_train.shape[1], X_train.shape[2]))

    # Fit the model and capture the history
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_data=(X_test, y_test), callbacks=[callback])
    return model, history

def save_model(model, file_path):
    """
    Saves the model to a file.

    Args:
        model: Trained model.
        file_path: File path to save the model.
    """
    model.save(file_path)

def load_model_from_file(file_path):
    """
    Loads the model from a file.

    Args:
        file_path: File path to load the model from.

    Returns:
        Loaded model.
    """
    return load_model(file_path)

class ProgressCallback(Callback):
    """
    Callback class for updating the progress.
    """
    def __init__(self, gui):
        super().__init__()
        self.gui = gui

    def on_epoch_end(self, epoch, logs=None):
        """
        Updates the progress at the end of each epoch.

        Args:
            epoch: The current epoch number.
            logs: Dict. Currently no data is passed to this argument for this method but that may change in the future.
        """
        loss = logs.get('loss')
        self.gui.losses.append(loss)  # Append the loss to gui's losses attribute
        self.gui.update_loss_plot(self.gui.losses, epoch)
        self.gui.update_progress((epoch + 1) / int(self.gui.epoch_spinbox.get()) * 100)
