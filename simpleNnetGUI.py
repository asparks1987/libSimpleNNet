import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import libSimpleNnet

class Application(tk.Frame):
    """
    GUI class, inherits from Tkinter's Frame class.
    """
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
        """
        Initialize the GUI widgets.
        """
        self.load_btn = tk.Button(self, text="LOAD TRAINING DATA", command=self.load_csv)
        self.load_btn.pack(side="top")

        self.epoch_label = tk.Label(self, text="Number of Epochs: ")
        self.epoch_label.pack(side="top")

        self.epoch_spinbox = tk.Spinbox(self, from_=1, to=1000)
        self.epoch_spinbox.pack(side="top")

        self.train_btn = tk.Button(self, text="TRAIN NETWORK", command=self.train_network)
        self.train_btn.pack(side="top")

        self.progress = ttk.Progressbar(self, length=200, mode='determinate')
        self.progress.pack(side="top")

        self.save_model_btn = tk.Button(self, text="SAVE MODEL", command=self.save_model)
        self.save_model_btn.pack(side="top")

        self.load_model_btn = tk.Button(self, text="LOAD MODEL", command=self.load_model)
        self.load_model_btn.pack(side="top")

        self.run_model_btn = tk.Button(self, text="RUN MODEL", command=self.run_model)
        self.run_model_btn.pack(side="top")

    def load_csv(self):
        """
        Load the CSV file using a file dialog and preprocess the data.
        """
        file_path = filedialog.askopenfilename()
        df = pd.read_csv(file_path)
        self.X_train, self.y_train = libSimpleNnet.preprocess_data(df, 'target')

    def update_progress(self, value):
        """
        Update the progress bar.
        """
        self.progress["value"] = value
        self.update_idletasks()

    def update_loss_plot(self, losses, epoch):
        """
        Update the loss plot.
        """
        self.loss_plot.clear()
        self.loss_plot.plot(range(epoch+1), self.losses[:epoch+1], 'r')  # Use self.losses directly
        self.canvas.draw()


    def train_network(self):
        """
        Train the LSTM model and update the progress bar and loss plot.
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            tk.messagebox.showerror("Error", "No data loaded for training. Please load a data file first.")
            return

        epochs = int(self.epoch_spinbox.get())
        self.losses = []  # Initialize losses before training
        callback = libSimpleNnet.ProgressCallback(self)
        self.model, self.history = libSimpleNnet.train_model(self.X_train, self.y_train, callback=callback, epochs=epochs)
        self.losses = self.history.history['loss']  # Update losses after training
    def update_epoch_progress(self, epoch):
        """
        Update the text widget to display the progress of each epoch.
        """
        self.progress_text.delete('1.0', tk.END)
        self.progress_text.insert(tk.END, f"Completed {epoch} out of {self.epoch_spinbox.get()} epochs")

    def save_model(self):
        """
        Save the trained model to a file.
        """
        if not hasattr(self, 'model'):
            tk.messagebox.showerror("Error", "No model has been trained yet.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".h5")
        if file_path:
            libSimpleNnet.save_model(self.model, file_path)

    def load_model(self):
        """
        Load a previously saved model from a file.
        """
        file_path = filedialog.askopenfilename(filetypes=(("HDF5 files", "*.h5"),))
        if not file_path:
            return
        try:
            self.model = libSimpleNnet.load_model_from_file(file_path)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to load model. Error: {str(e)}")

    def run_model(self):
        """
        Run the model on input data from the user.
        """
        if not hasattr(self, 'model'):
            tk.messagebox.showerror("Error", "No model has been loaded or trained yet.")
            return

        input_string = simpledialog.askstring("Input", "Enter your input data:")
        try:
            input_data = libSimpleNnet.preprocess_data_array(input_string)
            result = self.model.predict(input_data)
            tk.messagebox.showinfo("Model Output", f"The model's output is: {result[0]}")  # Assuming the output is a single value
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to run model. Error: {str(e)}")


root = tk.Tk()
app = Application(master=root)
app.mainloop()
