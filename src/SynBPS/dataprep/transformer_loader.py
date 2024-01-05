import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 

from ..constants import Task

class LogsDataLoader:
    def __init__(self, name, dir_path="./datasets"):
        """Initializes the LogsDataLoader with a dataset name and directory path.
        
        Args:
            name (str): The name of the dataset as used during processing raw logs.
            dir_path (str): Path to the directory where datasets are stored.
        """
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(self, df, x_word_dict, y_word_dict, max_case_length, shuffle=True):
        """Prepares data for the 'next activity' prediction task.
        
        Args:
            df (DataFrame): The dataframe containing the data.
            x_word_dict (dict): A dictionary mapping words to their integer indices.
            y_word_dict (dict): A dictionary mapping next activities to their integer indices.
            max_case_length (int): The maximum length of a case (sequence of events).
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            token_x: The tokenized and padded prefixes.
            token_y: The tokenized next activities.
        """
        # Extract the prefixes and next activities
        x = df["prefix"].values
        y = df["next_act"].values
        
        # Optionally shuffle the data
        if shuffle:
            x, y = utils.shuffle(x, y)

        # Tokenize the prefixes
        token_x = [[x_word_dict[s] for s in _x.split()] for _x in x]

        # Tokenize the next activities
        token_y = [y_word_dict[_y] for _y in y]

        # Pad the tokenized prefixes to the maximum case length
        token_x = tf.keras.preprocessing.sequence.pad_sequences(token_x, maxlen=max_case_length)

        # Convert to numpy arrays with type float32
        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y

    # The `prepare_data_next_time` and `prepare_data_remaining_time` methods are similar to
    # `prepare_data_next_activity` but focus on predicting the next time event and the remaining
    # time of the case, respectively. These methods also handle additional time-related features
    # and require different preprocessing steps, such as scaling the time features.

    def prepare_data_next_time(self, df, x_word_dict, max_case_length, time_scaler=None, y_scaler=None, shuffle=True):
        """
        Prepares training data for predicting the next time event in a log sequence.

        Parameters:
            df (DataFrame): The input dataframe with log data.
            x_word_dict (dict): A mapping from activities to integers (tokenization dictionary).
            max_case_length (int): The maximum length of a case for padding sequences.
            time_scaler (StandardScaler, optional): A scaler for standardizing time features.
            y_scaler (StandardScaler, optional): A scaler for standardizing the next time event.
            shuffle (bool): Whether to shuffle the dataset.

        Returns:
            np.array: Tokenized and padded sequences of activities (features).
            np.array: Time features associated with each sequence.
            np.array: The scaled next time events (labels).
            StandardScaler: The scaler used for time features.
            StandardScaler: The scaler used for the next time events.
        """
        # Extract sequences (prefixes), time features, and next time events from dataframe
        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32)
        y = df["next_time"].values.astype(np.float32)

        # Shuffle the data if required
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        # Tokenize the sequences
        token_x = [[x_word_dict[s] for s in _x.split()] for _x in x]

        # Scale the time features if a scaler is not provided, otherwise transform using the existing scaler
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)

        # Scale the next time events if a scaler is not provided, otherwise transform using the existing scaler
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        # Pad the tokenized sequences to ensure uniform length
        token_x = tf.keras.preprocessing.sequence.pad_sequences(token_x, maxlen=max_case_length)

        # Convert lists to numpy arrays with type float32 for compatibility with machine learning frameworks
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32).flatten()  # Flatten to ensure proper shape for labels

        return token_x, time_x, y, time_scaler, y_scaler
            
    def prepare_data_remaining_time(self, df, x_word_dict, max_case_length, time_scaler=None, y_scaler=None, shuffle=True): 
        # Extract prefixes and remaining time days from the dataframe 
        x = df["prefix"].values 
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(np.float32) 
        y = df["remaining_time_days"].values.astype(np.float32)

        # Shuffle the dataset if the shuffle flag is set to True
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        # Tokenize the prefixes using the provided word dictionary
        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        # Scale the time features if a scaler is not provided, otherwise use the existing scaler
        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)            

        # Scale the target variable (remaining time days) if a scaler is not provided, otherwise use the existing scaler
        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        # Pad the tokenized sequences to ensure they all have the same length
        token_x = tf.keras.preprocessing.sequence.pad_sequences(token_x, maxlen=max_case_length)

        # Convert the lists of tokens and time features to numpy arrays with type float32
        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Return the processed features, labels, and the scalers used for standardization
        return token_x, time_x, y, time_scaler, y_scaler
    

    def get_max_case_length(self, train_x):
         # Initialize a list to hold the number of tokens in each training case
        train_token_x = list()
        for _x in train_x:
            # Split the case into tokens and count them, then add the count to the list
            train_token_x.append(len(_x.split()))
        # Return the length of the longest case
        return max(train_token_x)

    def load_data(self, task):
        # Validate the task parameter to ensure it's one of the expected tasks
        if task not in (Task.NEXT_ACTIVITY,
            Task.NEXT_TIME,
            Task.REMAINING_TIME):
            raise ValueError("Invalid task.")
        
        # Load the training and testing datasets for the specified task from CSV files
        train_df = pd.read_csv(f"{self._dir_path}/{task.value}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task.value}_test.csv")

        # Open and read the metadata JSON file, which contains word dictionaries
        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        # Retrieve word dictionaries for input and output encoding from metadata
        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]

        # Use the previously defined get_max_case_length method to find the max length
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        # Determine the vocabulary size and total number of output classes
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, test_df, 
            x_word_dict, y_word_dict, 
            max_case_length, vocab_size, 
            total_classes)