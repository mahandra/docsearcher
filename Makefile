.PHONY: help up down build logs migrate makemigrations test lint fmt shell superuser

DC = docker compose
RUN = $(DC) run --rm web

help:  ## Show available commands
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

up:  ## Start db, redis and web (http://localhost:8000)
	$(DC) up -d --build

down:  ## Stop all services
	$(DC) down

build:  ## Rebuild the web image
	$(DC) build web

logs:  ## Follow web logs
	$(DC) logs -f web

migrate:  ## Apply database migrations
	$(RUN) python manage.py migrate

makemigrations:  ## Create new migrations
	$(RUN) python manage.py makemigrations

test:  ## Run the test suite (no real API calls)
	$(RUN) pytest $(ARGS)

lint:  ## Run ruff checks
	$(RUN) ruff check .

fmt:  ## Format code with ruff
	$(RUN) ruff format .

shell:  ## Open a Django shell
	$(RUN) python manage.py shell

superuser:  ## Create an admin user
	$(RUN) python manage.py createsuperuser
