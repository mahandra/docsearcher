from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    path("", views.conversation_list, name="conversation_list"),
    path("signup/", views.signup, name="signup"),
    path("c/new/", views.conversation_create, name="conversation_create"),
    path("c/<int:pk>/", views.conversation_detail, name="conversation_detail"),
    path("c/<int:pk>/delete/", views.conversation_delete, name="conversation_delete"),
    path("c/<int:pk>/messages/", views.message_create, name="message_create"),
    path("c/<int:pk>/stream/", views.message_stream, name="message_stream"),
]
