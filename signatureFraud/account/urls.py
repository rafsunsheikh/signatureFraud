from django.urls import path, include
from django.urls.resolvers import URLPattern
from django.conf import settings
from django.conf.urls.static import static
from . import views
from detection import views as detection_views

urlpatterns = [
    path("", views.index, name = "index"),
    path("check_signature", detection_views.check_signature, name = "check-signature"),
    path("add_signature", detection_views.add_signature, name = "add-signature"),
    path("remove_signature", detection_views.remove_signature, name = "remove-signature"),
    path("show_signature_list", detection_views.show_signature_list, name = "show-signature-list"),
]