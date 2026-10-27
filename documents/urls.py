from django.urls import path

from . import views

app_name = "documents"

urlpatterns = [
    path("", views.document_list, name="document_list"),
    path("search/", views.search_debug, name="search_debug"),
    path("upload/", views.document_upload, name="document_upload"),
    path("<int:pk>/", views.document_detail, name="document_detail"),
    path("<int:pk>/row/", views.document_row, name="document_row"),
    path("<int:pk>/delete/", views.document_delete, name="document_delete"),
    path("<int:pk>/reprocess/", views.document_reprocess, name="document_reprocess"),
]
