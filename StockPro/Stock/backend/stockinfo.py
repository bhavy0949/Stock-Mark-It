import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt, mpld3
import os
import pickle
import numpy as np

yf.pdr_override() 

def get_symbols():
    module_dir = os.path.dirname(__file__)  # get current directory
    file_path = os.path.join(module_dir, 'all_symbols.pkl')
    with open(file_path, 'rb') as file:
        symbols = pickle.load(file)
    return symbols

all_symbols = get_symbols()

def get_info(symbol):
    tick = yf.Ticker(symbol)
    hist =  tick.history(period="2d")
    stock = {}
    stock["symbol"] = symbol
    stock["name"] = next((s["name"] for s in all_symbols if s["symbol"] == symbol), None)
    stock["close"] = round(hist["Close"].tolist()[-1],2)
    stock["open"] = round(hist["Open"].tolist()[-1],2)
    stock["change"] = round(hist["Close"].tolist()[-1] - hist["Close"].tolist()[0],2)
    stock["pchange"] = round((stock["change"]/hist["Close"].tolist()[0])*100,2)
    if stock["change"] > 0:
        stock["color"] = "#00d600"
    else:
        stock["color"] = "red"
    stock["volume"] = hist["Volume"].tolist()[-1]
    return stock

def stock_today(symbol):
    df = pdr.get_data_yahoo(symbol, period="1d", interval="1m")
    df.to_csv('data.csv')
    df = pd.read_csv('data.csv')
    close = [round(x[0],2) for x in df.iloc[:, 4:5].astype('float32').values.tolist()]
    # TO DO

def get_stock():
    # .....
