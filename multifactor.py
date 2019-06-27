from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR) 
# Set to INFO for tracking training, default is WARN 

df = pd.read_csv("new_stock.csv")
temp_list = list(df.columns)


for i in range(0,5):
    temp_list.pop(0)

print(temp_list)

print("Using TensorFlow version %s" % (tf.__version__))

COLUMNS = temp_list

df = pd.read_csv("new_stock.csv", header=None, names=COLUMNS)
#print(df.head())



BATCH_SIZE = 40

def generate_input_fn(filename, batch_size=BATCH_SIZE):
    def _input_fn():
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TextLineReader()
        # Reads out batch_size number of lines
        key, rows = reader.read_up_to(filename_queue,
                                       num_records=batch_size)

        # record_defaults should match the datatypes of each respective column.
        record_defaults = []
        for i in len(COLUMNS):
            record_defaults.append([0])

#         rows = tf.Print(rows, [tf.shape(rows)], 'rows before expanding')

        # changes from shape = [Batch_size] to shape = [Batch_size, 1]
        rows = tf.expand_dims(rows, axis=-1)
#         rows = rows[:, np.newaxis]

#         rows = tf.Print(rows, [tf.shape(rows)], 'rows after expanding')

        # Decode CSV data that was just read out.
        columns = tf.decode_csv(
            rows, record_defaults=record_defaults)

        # features is a dictionary that maps from column names to tensors of the data.
        # income_bracket is the last column of the data. Note that this is NOT a dict.
        all_columns = dict(zip(COLUMNS, columns))

        # Save the income_bracket column as our labels
        # dict.pop() returns the popped array of income_bracket values
        income_bracket = all_columns.pop('income_bracket')

        # remove the fnlwgt key, which is not used
        all_columns.pop('fnlwgt', 'fnlwgt key not found')

        # the remaining columns are our features
        features = all_columns

        # Convert ">50K" to 1, and "<=50K" to 0
        labels = tf.to_int32(tf.equal(income_bracket, " >50K"))

        return features, labels

    return _input_fn

print('input function configured')



# The layers module contains many utilities for creating feature columns.
from tensorflow.contrib import layers

# Sparse base columns.
gender = layers.sparse_column_with_keys(column_name="gender",
                                        keys=["female", "male"])
race = layers.sparse_column_with_keys(column_name="race",
                                      keys=["Amer-Indian-Eskimo",
                                            "Asian-Pac-Islander",
                                            "Black", "Other",
                                            "White"])

education = layers.sparse_column_with_hash_bucket(
  "education", hash_bucket_size=1000)
marital_status = layers.sparse_column_with_hash_bucket(
  "marital_status", hash_bucket_size=100)
relationship = layers.sparse_column_with_hash_bucket(
  "relationship", hash_bucket_size=100)
workclass = layers.sparse_column_with_hash_bucket(
  "workclass", hash_bucket_size=100)
occupation = layers.sparse_column_with_hash_bucket(
  "occupation", hash_bucket_size=1000)
native_country = layers.sparse_column_with_hash_bucket(
  "native_country", hash_bucket_size=1000)

print('Sparse columns configured')


