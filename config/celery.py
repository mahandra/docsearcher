"""Celery application. Tasks live in <app>/tasks.py and are discovered automatically."""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("docqa")
# All CELERY_* Django settings configure Celery (e.g. CELERY_BROKER_URL -> broker_url).
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
