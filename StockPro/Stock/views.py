from Stock.backend.predict import predict_stock
from django.shortcuts import render, redirect
from django.http import JsonResponse
from Stock.backend.stockinfo import *
from Stock.backend.predict import *
from django.contrib.auth import login,logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
import sys
import time

def index(request):
    return render(request, "index.html")


def aboutus(request):
    return render(request, "aboutus.html")


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

def predict(request):
    data = None
    sym = ""
    if request.method == "POST":
        sym = request.POST["symbol"]
        data = request.POST["plot"]
    context = {
        "symbols": all_symbols,
        "data": data,
        "sym": sym
    }
    return render(request, "predict.html", context)

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        User(username=username, password=password).save()
        return redirect('/')
    else:
        return render(request, 'register.html')


def logout_request(request):
    logout(request)
    messages.info(request, "You have successfully logged out.") 
    return redirect("/accounts/login")

# def login_request(request):
# 	if request.method == "POST":
# 		form = AuthenticationForm(request, data=request.POST)
# 		if form.is_valid():
# 			username = form.cleaned_data.get('username')
# 			password = form.cleaned_data.get('password')
# 			user = authenticate(username=username, password=password)
# 			if user is not None:
# 				login(request, user)
# 				messages.info(request, f"You are now logged in as {username}.")
# 				return redirect("/")
# 			else:
# 				messages.error(request,"Invalid username or password.")
# 		else:
# 			messages.error(request,"Invalid username or password.")
# 	form = AuthenticationForm()
	# return render(request=request, template_name="../templates/registration/login.html", context={"login_form":form})