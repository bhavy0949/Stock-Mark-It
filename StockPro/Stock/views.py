from backend.predict import predict_stock
from django.shortcuts import render
from django.http import JsonResponse

def stock_predict(request, symbol, period, sim, future):
    data = predict_stock(symbol, period, sim, future)
    return JsonResponse({"data": data})