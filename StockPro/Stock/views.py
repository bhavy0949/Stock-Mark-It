from django.shortcuts import render

# Create your views here.
def stock_predict(request, symbol, period, sim, future):
    data = predict_stock(symbol, period, sim, future)
    return JsonResponse({"data": data})