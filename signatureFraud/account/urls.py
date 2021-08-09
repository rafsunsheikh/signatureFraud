from django.urls import path
from django.urls.resolvers import URLPattern
from . import views

app_name = "account"

urlpatterns = [
    # path("", views.homepage, name = "homepage"),
    
    path("register", views.register_request, name = "register")
]