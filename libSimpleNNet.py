from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import socket
from threading import Thread
import traceback

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_data(df, target_col):
    df['input'] = df['input'].apply(lambda x: [float(i) for i in x.split(',')])
    input_df = pd.DataFrame(df['input'].to_list(), columns=[f'input_{i}' for i in range(len(df['input'][0]))])
    df = pd.concat([input_df, df.drop('input', axis=1)], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 'D')
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y

def train_model(X_train, y_train, callback=None, epochs=5, validation_split=0.5):
    X_train_values = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = create_model((X_train.shape[1], 1))
    callbacks = [callback] if callback else []
    history = model.fit(X_train_values, y_train, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
    return model, history

class ServerThread(Thread):
    def __init__(self, port, model):
        super().__init__()
        self.port = port
        self.model = model

    def run(self):
        s = socket.socket()
        s.bind(("", self.port))
        s.listen(1)
        print(f"Server started on port {self.port}. Waiting for connections...")
        while True:
            conn, addr = s.accept()
            print(f"Handshake achieved with {addr}")
            try:
                while True:
                    input_str = conn.recv(1024).decode('utf-8')
                    if not input_str:
                        break
                    input_list = [float(x) for x in input_str.split(',')]
                    output = self.model.predict([input_list])
                    conn.sendall(str(output[0][0]).encode('utf-8'))
            except Exception as e:
                print(f"Exception occurred: {e}")
                print(traceback.format_exc())
            finally:
                conn.close()
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import socket
from threading import Thread
import traceback

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_data(df, target_col):
    df['input'] = df['input'].apply(lambda x: [float(i) for i in x.split(',')])
    input_df = pd.DataFrame(df['input'].to_list(), columns=[f'input_{i}' for i in range(len(df['input'][0]))])
    df = pd.concat([input_df, df.drop('input', axis=1)], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = (df['timestamp'] - df['timestamp'].min()) / np.timedelta64(1, 'D')
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y

def train_model(X_train, y_train, callback=None, epochs=5, validation_split=0.5):
    X_train_values = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = create_model((X_train.shape[1], 1))
    callbacks = [callback] if callback else []
    history = model.fit(X_train_values, y_train, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
    return model, history

class ServerThread(Thread):
    def __init__(self, port, model):
        super().__init__()
        self.port = port
        self.model = model

    def run(self):
        s = socket.socket()
        s.bind(("", self.port))
        s.listen(1)
        print(f"Server started on port {self.port}. Waiting for connections...")
        while True:
            conn, addr = s.accept()
            print(f"Handshake achieved with {addr}")
            try:
                while True:
                    input_str = conn.recv(1024).decode('utf-8')
                    if not input_str:
                        break
                    input_list = [float(x) for x in input_str.split(',')]
                    output = self.model.predict([input_list])
                    conn.sendall(str(output[0][0]).encode('utf-8'))
            except Exception as e:
                print(f"Exception occurred: {e}")
                print(traceback.format_exc())
            finally:
                conn.close()
