import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import zipfile

import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.losses import Huber
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.optimizers import adam_v2

def get_past(df):
    # Get old but not that old data
    return df.loc[('2019-04-01' <= df['Date']) & (df['Date'] <= '2019-05-01')]

def get_future(df):
  return df.loc[('2019-05-01' <= df['Date']) & (df['Date'] >= '2019-06-01')]

def split_data(all_dfs, n_stocks, n_stocks_val):
  # There is one test/eval data point for each train data point
  perm = np.random.permutation(n_stocks)
  val_stock_inds = perm[:n_stocks_val]
  test_stock_inds = perm[n_stocks_val: 2 * n_stocks_val]

  x_train = [get_future(df) for df in all_dfs[n_stocks_val * 2:]]
  x_val = [get_future(df) for df in all_dfs[val_stock_inds]]
  x_test = [get_future(df) for df in all_dfs[test_stock_inds]]
  return x_train, x_val, x_test

def normalize(df, sequence_length):
  volume = df['Volume'].iloc[1:]
  other = df.drop('Volume', axis=1)
  # Normalize everything but volume
  other += 1e-3
  normalized = np.log(other.shift(1)/other)[1:]
  normalized = normalized.fillna(0) # Remove NaN
  normalized = normalized / .03 # (normalized.std(axis=0)+ 1e-3)

  if normalized.isna().values.any():
    return None  # idk why this can happen

  # Normalize volume
  volume = (volume - np.min(volume)) / (np.max(volume) + 1) # not a great way to normalize
  volume = volume.fillna(0)
  normalized['Volume'] = volume

  assert not normalized.isna().values.any()
  SL = sequence_length + 1
  cut_off = normalized.tail(SL)
  rows_to_prepend = SL - cut_off.shape[0]
  if rows_to_prepend != 0:
    if rows_to_prepend == SL:
      return None
    base = [[0 for _ in range(cut_off.shape[1])] for _ in range(rows_to_prepend)]
    df_prepend = pd.DataFrame(base, columns=df.columns)
    cut_off = pd.concat([df_prepend, cut_off], ignore_index=True)
  return cut_off

def preprocess_data(raw_data, sequence_length):
  normalized = [normalize(df.drop("Date", axis=1), sequence_length) for df in raw_data]
  # We clip for some numerical stability, but it may make some
  # answers wrong/ easier to predict.
  filtered = [df.clip(-5, 5) for df in normalized if df is not None]
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

def get_data(sequence_length, n_stocks_train=1000, n_stocks_val=50, get_close_val_only=True):
    np.random.seed(1)
    TOTAL_STOCKS_ORIGINAL = min(len(os.listdir('data/archive/stocks')), n_stocks_train + 2 * n_stocks_val)
    N_STOCKS = TOTAL_STOCKS_ORIGINAL
    stock_paths = np.random.permutation(os.listdir('data/archive/stocks'))[:N_STOCKS]

    all_dfs = []
    for csvname in tqdm(stock_paths):
      all_dfs.append(pd.read_csv(os.path.join('data/archive/stocks', csvname)))
    all_dfs = np.array(all_dfs, dtype=object)
    print("Preprocessing Data")

    raw_train_data, raw_val_data, raw_test_data = split_data(all_dfs, N_STOCKS, n_stocks_val)
    for df in raw_train_data:
      assert isinstance(df, pd.DataFrame)


    x_train, y_train = preprocess_data(raw_train_data, sequence_length)
    x_val, y_val = preprocess_data(raw_val_data, sequence_length)
    x_test, y_test = preprocess_data(raw_test_data, sequence_length)

    assert x_train[17].shape == (sequence_length, 6)
    assert y_train[17].shape == (sequence_length,)

    # Shapes correct
    assert(x_train[0].shape[1] == 6)
    assert(x_train[17].shape[0] == y_train[17].shape[0])
    # Each output is passed to next input
    # assert x_train[0][1, 3] == y_train[0][0]
    # assert x_train[0][2, 3] == y_train[0][1]
    # assert x_train[0][3, 3] == y_train[0][2]
    assert not np.isnan(x_train[0]).any()
    if get_close_val_only:
        close_ind = 3
        def get_close_only(x_data):
            return [x_datapoint[:, close_ind] for x_datapoint in x_data]
        x_train = get_close_only(x_train)
        x_val = get_close_only(x_val)
        x_test = get_close_only(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def bounded_map(x):
    return np.arctan(x)

def inv_bounded_map(x):
    return np.tan(x)

def make_lstm(sequence_length, input_dim, num_hidden) -> Model:
    input = Input(shape=(sequence_length, input_dim))
    lstm = LSTM(num_hidden, dropout=.3)(input)
    logits = Dense(1, activation=None)(lstm)
    model = Model(input, logits)
    model.compile(optimizer=adam_v2.Adam(1.5e-3),
              loss=Huber(),
              metrics=['mse'])
    model.summary()
    return model

def train_model(model: Model, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset):
    epochs = 30
    tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir='tf_logs'),
      tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=5,
        restore_best_weights=True
      ),
      tf.keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints',
        save_weights_only=True,
        monitor='val_mse',
        mode='min',
        save_best_only=True)
      ]
    model.fit(
        train_dataset.batch(32),
        validation_data=val_dataset.batch(32),
        epochs=epochs,
        callbacks=[tensorboard_callback]
        )
    return model

def contant_baseline(y_train, y_val, y_test):
    pred_value = 0
    for y in y_train:
        pred_value += y[-1]
    pred_value /= len(y_train)
    print(pred_value)

    for name, test_set in ("train", y_train), ("Val", y_val), ("Test", y_test):
        losses = []
        print(f"N datapoints: {len(test_set)}")
        for y in test_set:
            losses.append((y[-1] - pred_value) ** 2)
        print(f"{name} Loss: {sum(losses) / len(losses)}")
    return pred_value

def tf_baseline(num_hidden, sequence_length, x_train, y_train, x_val, y_val, x_test, y_test):
    num_hidden = 64
    model_tf = make_lstm(sequence_length, 1, num_hidden)
    # Just use the last element of the time series
    y_train = [y[-1] for y in  y_train]
    y_val = [y[-1] for y in  y_val]
    y_test = [y[-1] for y in  y_test]
    train_tf = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_tf = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    model_tf = train_model(model_tf, train_tf, val_tf)
    test_tf = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    print("Train result:")
    model_tf.evaluate(train_tf.batch(32))
    print("Val result:")
    model_tf.evaluate(val_tf.batch(32))
    print("Test result:")
    model_tf.evaluate(test_tf.batch(32))
    
if __name__ == "__main__":
    assert abs(inv_bounded_map(bounded_map(.8543)) - .8543) < 1e-6
    assert abs(inv_bounded_map(bounded_map(-.8543)) + .8543) < 1e-6
    sequence_length = 5
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(sequence_length, 100, 64)
    print(np.mean(y_train))
    print(np.mean(y_val))
    print(np.mean(y_test))
    print(np.std(y_train))
    print(np.std(y_val))
    print(np.std(y_test))
    contant_baseline(y_train, y_val, y_test)
    tf_baseline(64, sequence_length,  x_train, y_train, x_val, y_val, x_test, y_test)
