import os
import csv
import sys
import logging
import random
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import zipfile
SEQUENCE_LENGTH = 10
def get_past(df):
    return df.loc[df['Date'] <= '2019-03-31']

# Note: We reserve the first 30 days to initalize the RNN,
# but do not score the model on these days
def get_future(df):
  return df.loc[df['Date'] >= '2019-03-01']

def split_data(all_dfs, n_stocks):
  # There is one test/eval data point for each train data point
  val_count = n_stocks // 2
  perm = np.random.permutation(n_stocks)
  val_stock_inds = perm[:val_count]
  test_stock_inds = perm[val_count:]

  x_train = [get_past(df) for df in all_dfs]
  x_val = [df for df in all_dfs[val_stock_inds]]
  x_test = [df for df in all_dfs[test_stock_inds]]
  return x_train, x_val, x_test

def normalize(df):
  volume = df['Volume'][1:]
  other = df.drop('Volume', axis=1)
  # Normalize everything but volume
  other += 1e-3
  normalized = np.log(other.shift(1)/other)[1:]
  normalized = normalized.fillna(0) # Remove NaN
  normalized = normalized / (normalized.std(axis=0)+ 1e-3)

  if normalized.isna().values.any():
    print("FAIL2")
    return None  # idk why this can happen

  # Normalize volume
  volume = (volume - np.min(volume)) / (np.max(volume) + 1) # not a great way to normalize
  volume = volume.fillna(0)
  normalized['Volume'] = volume

  assert not normalized.isna().values.any()
  SL = SEQUENCE_LENGTH + 1
  cut_off = normalized.tail(SL)
  rows_to_prepend = SL - cut_off.shape[0]
  if rows_to_prepend != 0:
    if rows_to_prepend == SL:
      return None
    base = [[0 for _ in range(cut_off.shape[1])] for _ in range(rows_to_prepend)]
    df_prepend = pd.DataFrame(base, columns=df.columns)
    cut_off = df_prepend.append(cut_off, ignore_index=True)
  return cut_off
def preprocess_data(raw_data):
  normalized = [normalize(df.drop("Date", axis=1)) for df in raw_data]
  filtered = [df for df in normalized if df is not None]
  # x[1].close is y[0]
  x = [df[:-1] for df in filtered]
  y = [df["Close"].shift(-1).iloc[:-1] for df in filtered]
  out_x = []
  out_y = []
  for xe, ye in zip(x, y):
    if len(xe) != 0:
      out_x.append(np.array(xe, dtype=np.float32))
      out_y.append(np.array(ye, dtype=np.float32))
    assert not xe.isnull().values.any()
    assert not ye.isnull().values.any()

  return out_x, out_y
def extract_zip():
    print("Extracting zip")
    path_to_zip_file = "data/archive.zip"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall("data/archive")

def get_data():
    np.random.seed(0)
    print("Loading stocks")
    TOTAL_STOCKS_ORIGINAL = min(len(os.listdir('data/archive/stocks')), 50)
    N_STOCKS = TOTAL_STOCKS_ORIGINAL
    stock_paths = np.random.permutation(os.listdir('data/archive/stocks'))[:N_STOCKS]

    all_dfs = []
    for csvname in tqdm(stock_paths):
      all_dfs.append(pd.read_csv(os.path.join('data/archive/stocks', csvname)))
    all_dfs = np.array(all_dfs, dtype=object)
    print("Preprocessing Data")

    raw_train_data, raw_val_data, raw_test_data = split_data(all_dfs, N_STOCKS)
    for df in raw_train_data:
      assert isinstance(df, pd.DataFrame)


    x_train, y_train = preprocess_data(raw_train_data)
    x_val, y_val = preprocess_data(raw_val_data)
    x_test, y_test = preprocess_data(raw_test_data)

    assert x_train[17].shape == (SEQUENCE_LENGTH, 6)
    assert y_train[17].shape == (SEQUENCE_LENGTH,)
    
        # Shapes correct
    assert(x_train[0].shape[1] == 6)
    assert(x_train[17].shape[0] == y_train[17].shape[0])
    # Each output is passed to next input
    assert x_train[0][1, 3] == y_train[0][0]
    assert x_train[0][2, 3] == y_train[0][1]
    assert x_train[0][3, 3] == y_train[0][2]
    assert not np.isnan(x_train[0]).any()
    return x_train, y_train, x_val, y_val, x_test, y_test

def bounded_map(x):
    return np.tanh(x) * np.pi / 2

def inv_bounded_map(x):
    return np.arctanh(x * 2 / np.pi)

if __name__ == "__main__":
    assert abs(inv_bounded_map(bounded_map(.8543)) - .8543) < 1e-6
    get_data()