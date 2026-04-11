from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health_check, name="health_check"),
    path("", views.index, name="home"),
    path("index.html", views.index, name="index"),
    path("signup/", views.Register, name="Register"),
    path("Register.html", views.Register, name="RegisterLegacy"),
    path("RegisterAction", views.RegisterAction, name="RegisterAction"),
    path("login/", views.UserLogin, name="UserLogin"),
    path("UserLogin.html", views.UserLogin, name="UserLoginLegacy"),
    path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
    path("logout/", views.LogoutUser, name="LogoutUser"),
    path("dashboard/", views.Dashboard, name="Dashboard"),
    path("LoadDatasetAction", views.LoadDatasetAction, name="LoadDatasetAction"),
    path("ProcessData", views.ProcessData, name="ProcessData"),
    path("RunML", views.RunML, name="RunML"),
    path("Predict", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
]
