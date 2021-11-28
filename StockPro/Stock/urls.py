from django.urls import path

from . import views

urlpatterns = [
    path("about_us", views.aboutus, name="aboutus"),
    # path("login", views.login_request, name="login"),
    path("logout/", views.logout_request, name="logout"),
    path("register", views.register, name="register"),
    path("", views.index, name="index"),
    path("stock", views.stock, name="stock"),
    path("predict", views.predict, name="predict"),
    path("predict_stock/<str:symbol>/<str:period>/<int:sim>/<int:future>", views.stock_predict, name="predict_stock"),
]
