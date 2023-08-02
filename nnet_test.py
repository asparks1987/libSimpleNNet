import sys
import libSimpleNNet
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import subprocess
import matplotlib.pyplot as plt

# To run an external script
def run_external_script(script_path='libpairgenerator\generate.py'):
    subprocess.call(['python', script_path])


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Setting up the main window
        self.setWindowTitle("Neural Network GUI")
        self.setGeometry(100, 100, 800, 600)

        # Placeholder for loss history
        self.loss_history = []

        # Adding buttons
        self.load_train_button = QPushButton("Load Training Data", self)
        self.load_train_button.move(20, 20)
        self.load_train_button.clicked.connect(self.load_training_data)

        self.load_test_button = QPushButton("Load Test Data", self)
        self.load_test_button.move(20, 60)
        self.load_test_button.clicked.connect(self.load_test_data)

        self.save_network_button = QPushButton("Save Network", self)
        self.save_network_button.move(20, 100)
        self.save_network_button.clicked.connect(self.save_network)

        self.load_network_button = QPushButton("Load Network", self)
        self.load_network_button.move(20, 140)
        self.load_network_button.clicked.connect(self.load_network)

        self.run_network_button = QPushButton("Run Network", self)
        self.run_network_button.move(20, 180)
        self.run_network_button.clicked.connect(self.run_network)

        self.run_script_button = QPushButton("Run External Script", self)
        self.run_script_button.move(20, 220)
        self.run_script_button.clicked.connect(self.run_script)

        self.label = QLabel(self)
        self.label.move(20, 260)

        # Adding a plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_widget = QWidget(self)
        self.plot_widget.setGeometry(200, 20, 570, 550)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.plot_widget.setLayout(layout)

    # Function to load training data
    @pyqtSlot()
    def load_training_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open training data file", "", "CSV Files (*.csv)")
        # load your training data using the filename
        self.label.setText("Training data loaded from " + filename)
        
    # Function to load test data
    @pyqtSlot()
    def load_test_data(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open test data file", "", "CSV Files (*.csv)")
        # load your test data using the filename
        self.label.setText("Test data loaded from " + filename)

    # Function to save the network
    @pyqtSlot()
    def save_network(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save network", "", "H5 Files (*.h5)")
        # save your network to the filename
        self.label.setText("Network saved to " + filename)

    # Function to load the network
    @pyqtSlot()
    def load_network(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open network file", "", "H5 Files (*.h5)")
        # load your network from the filename
        self.label.setText("Network loaded from " + filename)

    # Function to run the network
    @pyqtSlot()
    def run_network(self):
        # run your network here, and update self.loss_history with the loss values
        self.label.setText("Running the network...")
        self.update_plot()

    # Function to run an external script
    @pyqtSlot()
    def run_script(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Python script", "", "Python Files (*.py)")
        self.label.setText("Running external script from " + filename)
        threading.Thread(target=run_external_script, args=(filename,)).start()

    # Function to update the plot
    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(self.loss_history)
        self.canvas.draw()


# Running the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
