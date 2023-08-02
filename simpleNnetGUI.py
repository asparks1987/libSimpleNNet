import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from keras.callbacks import Callback
from libSimpleNnet import preprocess_data, train_model, ServerThread

class ProgressCallback(Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        self.app.update_progress((epoch + 1) / self.params['epochs'] * 100)
        self.app.update_loss_plot(logs['loss'], epoch)

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

    def create_widgets(self):
        self.load_btn = tk.Button(self, text="LOAD TRAINING DATA", command=self.load_data)
        self.load_btn.pack(side="top")

        self.train_btn = tk.Button(self, text="TRAIN NETWORK", command=self.train_network)
        self.train_btn.pack(side="top")

        self.run_btn = tk.Button(self, text="RUN SERVER", command=self.run_server)
        self.run_btn.pack(side="top")

        self.progress = ttk.Progressbar(self, length=200, mode='determinate')
        self.progress.pack(side="top")

    def load_data(self):
        self.csv_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        df = pd.read_csv(self.csv_file)
        self.X_train, self.y_train = preprocess_data(df, 'target')

    def update_progress(self, value):
        self.progress["value"] = value
        self.update_idletasks()

    def update_loss_plot(self, loss, epoch):
        self.loss_plot.plot(range(epoch+1), loss, 'r')
        self.canvas.draw()

    def train_network(self):
        self.progress["value"] = 0
        self.update_idletasks()
        input_shape = (self.X_train.shape[1], 1)
        callback = ProgressCallback(self)
        self.model, self.history = train_model(self.X_train, self.y_train, callback=callback)
        self.progress["value"] = 100
        self.update_idletasks()

    def run_server(self):
        port = int(input("Enter the port to listen on: "))
        server_thread = ServerThread(port, self.model)
        server_thread.start()

def run():
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
