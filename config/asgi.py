"""ASGI entry point (served by uvicorn)."""

import os

from django.conf import settings
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_asgi_application()

if settings.DEBUG:
    # runserver serves static files automatically; under uvicorn we do it explicitly.
    from django.contrib.staticfiles.handlers import ASGIStaticFilesHandler

    application = ASGIStaticFilesHandler(application)
