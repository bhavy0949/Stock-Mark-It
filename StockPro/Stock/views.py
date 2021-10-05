from Stock.backend.predict import predict_stock
from django.shortcuts import render
from django.http import JsonResponse
from Stock.backend.stockinfo import *
from Stock.backend.predict import *
import sys
import time

def index(request):
    return render(request, "index.html")


def stock_predict(request, symbol, period, sim, future):
    data = predict_stock(symbol, period, sim, future)
    return JsonResponse({"data": data})

def stock(request):
    data = None
    stock = None
    if request.method == "POST":
        sym = request.POST["symbol"]
        time = request.POST["period"]
        if time=="1d":
            data = stock_today(sym)
        else:
            data = get_stock(sym, time)
        stock = get_info(sym)
    context = {
        "data": data,
        "symbols": all_symbols,
        "stock": stock
    }
    return render(request, "stock.html", context)

