# Time Series Prediction with LSTM

This script uses a Long Short-Term Memory (LSTM) model to predict future values in a time series. It demonstrates the complete machine learning workflow including data loading, preprocessing, model creation, training, evaluation, prediction, and model persistence.

## Installation

Before running the script, make sure to install the necessary Python libraries:

```bash
pip install pandas numpy scikit-learn keras tensorflow

Sure, here it is as a markdown text:

markdown
Copy code
# Time Series Prediction with LSTM

This script uses a Long Short-Term Memory (LSTM) model to predict future values in a time series. It demonstrates the complete machine learning workflow including data loading, preprocessing, model creation, training, evaluation, prediction, and model persistence.

## Installation

Before running the script, make sure to install the necessary Python libraries:

```bash
pip install pandas numpy scikit-learn keras tensorflow
Script Usage
Data Format
This script requires a CSV file as input. The CSV file should include a date and time column, as well as a column for the target variable to predict. The remaining columns are considered as features.

Here's an example of the expected CSV file format:
date,time,feature1,feature2,target
2021-01-01,00:00:00,123.4,567.8,9.10
2021-01-01,01:00:00,234.5,678.9,0.12
Function Overview
The script contains the following functions:

load_data(file_path: str) -> pd.DataFrame: Loads data from a CSV file.
preprocess_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]: Preprocesses the data by converting the date and time to timestamp, normalizing all feature columns, and separating the features and target variable.
split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Splits the data into training and test sets.
create_model(input_shape: Tuple[int, int]) -> keras.models.Sequential: Creates a LSTM model.
train_model(model: keras.models.Sequential, X_train: pd.DataFrame, y_train: pd.Series, epochs: int = 50, validation_split: float = 0.2) -> Tuple[keras.models.Sequential, keras.callbacks.History]: Trains the LSTM model on the training data.
evaluate_model(model: keras.models.Sequential, X_test: pd.DataFrame, y_test: pd.Series) -> float: Evaluates the model on the test data and returns the loss.
predict_sample(model: keras.models.Sequential, X_test: pd.DataFrame) -> float: Predicts the value for the first sample in the test set.
save_model(model: keras.models.Sequential, model_path: str): Saves the trained model to a file.
load_saved_model(model_path: str) -> keras.models.Sequential: Loads a trained model from a file.
By default, the script reads the data from a file named data.csv, uses a column named target as the target variable, trains a LSTM model, and saves the trained model to a file named model.h5.

You can modify these defaults by changing the following lines in the script:
file_path = 'data.csv'
target_col = 'target'
model_path = 'model.h5'
Error Handling
The script includes basic error handling. If an error occurs (e.g., the data file does not exist, the target column is not found in the data, etc.), the script will print an error message and terminate.

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

License
MIT