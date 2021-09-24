import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

# import tensorflow as tf
# import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sns.set()
tf.compat.v1.random.set_random_seed(1234)
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

num_layers = 1
size_layer = 128
timestamp = 10
epoch = 300
dropout_rate = 0.8
test_size = 30
learning_rate = 0.01


def predict_stock(symbol, period, sim, future):
    simulation_size = sim
    test_size = future
    # download dataframe using pandas_datareader
    df = pdr.get_data_yahoo(symbol, period=period, interval="1d")
    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')
    minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
    df_log = pd.DataFrame(df_log)

    df_train = df_log
    class Model:
        def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias = 0.1,
        ):
            def lstm_cell(size_layer):
                return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

            rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)],
                state_is_tuple = False,
            )
            self.X = tf.placeholder(tf.float32, (None, None, size))
            self.Y = tf.placeholder(tf.float32, (None, output_size))
            drop = tf.contrib.rnn.DropoutWrapper(
                rnn_cells, output_keep_prob = forget_bias
            )
            self.hidden_layer = tf.placeholder(
                tf.float32, (None, num_layers * 2 * size_layer)
            )
            self.outputs, self.last_state = tf.nn.dynamic_rnn(
                drop, self.X, initial_state = self.hidden_layer, dtype = tf.float32
            )
            self.logits = tf.layers.dense(self.outputs[-1], output_size)
            self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.cost
            )
        