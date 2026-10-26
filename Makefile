.PHONY: help up down build logs migrate makemigrations test lint fmt shell superuser

DC = docker compose
# Run as the host user so files created in the container (migrations) are not owned by root.
RUN = $(DC) run --rm --user $(shell id -u):$(shell id -g) web

help:  ## Show available commands
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

up:  ## Start db, redis and web (http://localhost:8000)
	$(DC) up -d --build

down:  ## Stop all services
	$(DC) down

build:  ## Rebuild the web image
	$(DC) build web

logs:  ## Follow web and worker logs
	$(DC) logs -f web worker

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
