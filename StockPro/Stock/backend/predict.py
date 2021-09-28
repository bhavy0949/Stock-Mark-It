import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()

test_size = 30


def predict_stock(symbol, period, sim, future):

    df = pdr.get_data_yahoo(symbol, period=period, interval="1d")
    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')
   