from django.shortcuts import render, redirect
from .models import User
# Create your views here.
def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        User(username=username, password=password).save()
        return redirect('/')
    else:
        return render(request, 'register.html')