# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a csv file and returns a DataFrame.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    pd.DataFrame: Loaded data

    Raises:
    FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def preprocess_data(df, target_col):
    """
    This function preprocesses the DataFrame by converting the time and date to timestamp and normalizing all columns.
    It also splits the data into input features and target variable.
    """
    if target_col not in df.columns:
        raise Exception(f"Target column not found in DataFrame: {target_col}")
        
    # Convert the time and date to timestamp
    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 'D')

    # Split data into input features and target variable
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def split_data(X, y):
    """
    This function splits the data into training and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_model(input_shape):
    """
    This function creates and returns a LSTM model with the specified input shape.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(model, X_train, y_train, epochs=50, validation_split=0.2):
    """
    This function trains the model on the training data.
    """
    X_train_values = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    history = model.fit(X_train_values, y_train, epochs=epochs, validation_split=validation_split)
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    This function evaluates the model on the test data and returns the loss.
    """
    X_test_values = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    loss = model.evaluate(X_test_values, y_test)
    return loss

def predict_sample(model, X_test):
    """
    This function predicts the value for the first sample in the test set.
    """
    sample = X_test.iloc[0]  # take the first sample from the test set
    sample_reshaped = sample.values.reshape((1, sample.shape[0], 1))
    prediction = model.predict(sample_reshaped)
    return prediction[0][0]

def save_model(model, model_path):
    """
    This function saves the trained model to a file.
    """
    model.save(model_path)

def load_saved_model(model_path):
    """
    This function loads a trained model from a file.
    """
    model = load_model(model_path)
    return model

# The main code execution block
if __name__ == "__main__":
    file_path = 'data.csv'
    target_col = 'target'
    model_path = 'model.h5'

    try:
        df = load_data(file_path)
        X, y = preprocess_data(df, target_col)
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = create_model(input_shape=(X_train.shape[1], 1))
        model, _ = train_model(model, X_train, y_train)
        save_model(model, model_path)
        model = load_saved_model(model_path)
        loss = evaluate_model(model, X_test, y_test)
        print('Test loss:', loss)
        prediction = predict_sample(model, X_test)
        print('Predicted value:', prediction)
    except Exception as e:
        print(f"An error occurred: {str(e)}")