.PHONY: build-and-push test lint

# Переменные
DOCKER_USERNAME ?= shrtcve
IMAGE_NAME := ml-service
IMAGE_TAG := latest
REGISTRY := $(DOCKER_USERNAME)

# Сборка образа и пуш в DockerHub
build-and-push:
	@echo "Building Docker image"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "Tagging image for registry"
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "Pushing to DockerHub"
	docker push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "build and puch done"

# пайтесты
test:
	@echo "tests start"
	poetry run pytest tests/ -v --cov=app --cov-report=html --cov-report=term
	@echo "tests done"

# линтер + автофиксер
lint-fix:
	@echo "auto-fixing stuff"
	poetry run ruff check --fix app/ tests/
	poetry run ruff format app/ tests/
	@echo "stuff formatted"

help:
	@echo "Available commands:"
	@echo "make build-and-push - Build Docker image and push to DockerHub"
	@echo "make test - Run unit tests with coverage"
	@echo "make lint - Run linters"
	@echo "make help - Show this help message (how do you know this before writing help is a great mystery)"
