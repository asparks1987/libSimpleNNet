import sys
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, scrolledtext
import numpy as np
import socket
import threading

class OutputRedirector:
    def __init__(self, app):
        self.app = app

    def write(self, string):
        self.app.console_insert(string)

    def flush(self):
        pass

class ProgressCallback(Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        self.app.update_progress((epoch + 1) / self.params['epochs'] * 100)
        self.app.losses.append(logs['loss'])
        self.app.update_loss_plot()

def preprocess_data(df, target_col):
    if target_col not in df.columns:
        raise Exception(f"Target column not found in DataFrame: {target_col}")

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

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(X_train, y_train, callback=None, epochs=5, validation_split=0.5):
    X_train_values = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = create_model((X_train.shape[1], 1))

    callbacks = [callback] if callback else []

    history = model.fit(X_train_values, y_train, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

    return model, history

class ServerThread(threading.Thread):
    def __init__(self, app, port):
        super().__init__()
        self.app = app
        self.port = port

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', self.port))
        print(f"Server started at localhost:{self.port}\n")
        s.listen(1)
        conn, addr = s.accept()
        print("Connection established with client\n")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            inputs = np.fromstring(data.decode(), sep=',')
            inputs = inputs.reshape((1, len(inputs), 1))
            outputs = self.app.model.predict(inputs)
            conn.sendall(str(outputs[0][0]).encode())
        conn.close()
        print("Connection closed\n")

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.loss_plot = self.figure.add_subplot(111)
        self.loss_plot.set_xlabel("Epoch")
        self.loss_plot.set_ylabel("Loss")
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack()
        self.losses = []

    def create_widgets(self):
        self.load_btn = tk.Button(self, text="LOAD TRAINING DATA", command=self.load_data)
        self.load_btn.pack(side="top")

        self.train_btn = tk.Button(self, text="TRAIN NETWORK", command=self.train_network)
        self.train_btn.pack(side="top")

        self.run_btn = tk.Button(self, text="RUN", command=self.run_server)
        self.run_btn.pack(side="top")

        self.progress = ttk.Progressbar(self, length=200, mode='determinate')
        self.progress.pack(side="top")

        self.console = scrolledtext.ScrolledText(self, height=10)
        self.console.pack(side="top")

    def load_data(self):
        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        df = pd.read_csv(self.csv_file)
        self.X_train, self.y_train = preprocess_data(df, 'target')

    def update_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    def update_loss_plot(self):
        self.loss_plot.clear()
        self.loss_plot.plot(self.losses)
        self.canvas.draw()

    def console_insert(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def train_network(self):
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise Exception("No data loaded for training. Please load a data file first.")

        self.progress["value"] = 0
        self.update_idletasks()

        input_shape = (self.X_train.shape[1], 1)
        model = create_model(input_shape)
        callback = ProgressCallback(self)

        self.model, self.history = train_model(self.X_train, self.y_train, callback=callback)

        self.progress["value"] = 100
        self.update_idletasks()

    def run_server(self):
        port = simpledialog.askinteger("Port", "Enter port number")
        if port:
            server_thread = ServerThread(self, port)
            server_thread.start()

root = tk.Tk()
app = Application(master=root)
sys.stdout = OutputRedirector(app)
app.mainloop()
