"""Django settings for DocQA.

All environment-specific values come from environment variables (see .env.example).
"""

from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env()
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("DJANGO_SECRET_KEY")
DEBUG = env.bool("DJANGO_DEBUG", default=False)
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=["localhost", "127.0.0.1"])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "chat",
    "documents",
    "evals",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ASGI only: streaming LLM responses hold a connection open for tens of seconds,
# which would pin a whole worker under WSGI.
ASGI_APPLICATION = "config.asgi.application"

DATABASES = {"default": env.db("DATABASE_URL")}
# Persistent connections are not safe with async views (connections are per-thread
# and async code hops between threads), so keep the Django default of 0.
DATABASES["default"]["CONN_MAX_AGE"] = 0

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "chat:conversation_list"
LOGOUT_REDIRECT_URL = "login"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Uploaded files; shared between the web and worker containers.
MEDIA_URL = "media/"
MEDIA_ROOT = env.path("MEDIA_ROOT", default=BASE_DIR / "media")

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- LLM -------------------------------------------------------------------------
# The Anthropic SDK reads ANTHROPIC_API_KEY from the environment by itself;
# we only validate here that it is present when the app needs it (see chat/llm.py).
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY", default="")
LLM_MODEL = env("LLM_MODEL", default="claude-opus-5-5")
# Effort trades answer quality for latency/tokens: low | medium | high | xhigh | max.
LLM_EFFORT = env("LLM_EFFORT", default="low")
LLM_MAX_TOKENS = env.int("LLM_MAX_TOKENS", default=16000)
# Only the last N messages are sent to the model, to bound per-request cost.
LLM_MAX_HISTORY_MESSAGES = env.int("LLM_MAX_HISTORY_MESSAGES", default=20)
# SDK-level retries with exponential backoff (429, 5xx, 529 overloaded, network).
LLM_MAX_RETRIES = env.int("LLM_MAX_RETRIES", default=3)
LLM_TIMEOUT_SECONDS = env.float("LLM_TIMEOUT_SECONDS", default=120.0)
# Prompt caching: the system prompt and the conversation history are cached for 5 minutes,
# so the next turn re-reads them at a fraction of the input price.
LLM_PROMPT_CACHING = env.bool("LLM_PROMPT_CACHING", default=True)

# Langfuse tracing (chat/tracing.py): off unless both keys are set.
LANGFUSE_PUBLIC_KEY = env("LANGFUSE_PUBLIC_KEY", default="")
LANGFUSE_SECRET_KEY = env("LANGFUSE_SECRET_KEY", default="")
LANGFUSE_BASE_URL = env("LANGFUSE_BASE_URL", default="https://cloud.langfuse.com")
LANGFUSE_ENVIRONMENT = env("LANGFUSE_ENVIRONMENT", default="development")
# Questions, retrieved passages and answers are user data sent to a third party.
LANGFUSE_CAPTURE_CONTENT = env.bool("LANGFUSE_CAPTURE_CONTENT", default=True)

# --- Celery ---------------------------------------------------------------------------
CELERY_BROKER_URL = env("REDIS_URL", default="redis://localhost:6380/0")
# Task outcomes are stored on the Document itself, so no result backend is needed.
CELERY_TASK_IGNORE_RESULT = True
# Acknowledge a task only after it finished: if the worker dies mid-task, the task is
# redelivered. Requires idempotent tasks.
CELERY_TASK_ACKS_LATE = True
CELERY_TASK_REJECT_ON_WORKER_LOST = True
# Take one task at a time; ingestion tasks are long, so prefetching would starve others.
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_SOFT_TIME_LIMIT = 540
CELERY_TASK_TIME_LIMIT = 600
# Keep our LOGGING configuration instead of Celery's.
CELERY_WORKER_HIJACK_ROOT_LOGGER = False
# Run tasks synchronously in-process (tests, debugging without a worker).
CELERY_TASK_ALWAYS_EAGER = env.bool("CELERY_TASK_ALWAYS_EAGER", default=False)

# --- Documents & embeddings ---------------------------------------------------------
# Checked in the upload form. In production also cap the body size at the proxy
# (e.g. nginx client_max_body_size): Django's DATA_UPLOAD_MAX_MEMORY_SIZE ignores files.
DOCUMENT_MAX_UPLOAD_MB = env.int("DOCUMENT_MAX_UPLOAD_MB", default=20)

VOYAGE_API_KEY = env("VOYAGE_API_KEY", default="")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", default="voyage-4")
EMBEDDING_BATCH_SIZE = env.int("EMBEDDING_BATCH_SIZE", default=128)
EMBEDDING_MAX_RETRIES = env.int("EMBEDDING_MAX_RETRIES", default=3)
EMBEDDING_TIMEOUT_SECONDS = env.float("EMBEDDING_TIMEOUT_SECONDS", default=60.0)

# Chunk size and overlap in (estimated) tokens.
CHUNK_SIZE_TOKENS = env.int("CHUNK_SIZE_TOKENS", default=500)
CHUNK_OVERLAP_TOKENS = env.int("CHUNK_OVERLAP_TOKENS", default=75)

# Retrieval: each search (vector, keyword) returns RETRIEVAL_CANDIDATES chunks; after
# fusion the best RETRIEVAL_TOP_K go into the prompt.
RETRIEVAL_TOP_K = env.int("RETRIEVAL_TOP_K", default=8)
RETRIEVAL_CANDIDATES = env.int("RETRIEVAL_CANDIDATES", default=20)
# RRF smoothing constant: smaller = the top ranks of each search weigh more. Tuned with
# `manage.py eval_retrieval` (k=60 from the RRF paper lost cross-lingual hits).
RETRIEVAL_RRF_K = env.int("RETRIEVAL_RRF_K", default=5)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "{asctime} {levelname} {name}: {message}", "style": "{"},
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "chat": {"level": env("CHAT_LOG_LEVEL", default="INFO")},
        "documents": {"level": env("DOCUMENTS_LOG_LEVEL", default="INFO")},
        "celery": {"level": "INFO"},
        # Override uvicorn's own handlers so server and access logs share our format.
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
    },
}
