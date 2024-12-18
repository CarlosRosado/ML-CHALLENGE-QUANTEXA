# Variables
DOCKER_USERNAME = carlosrosado
IMAGE_NAME = ml-challenge-quantexa-web
TAG = latest
NAMESPACE = default
DEPLOYMENT_NAME = quantexa-text-classifier-deployment
SERVICE_NAME = quantexa-text-classifier-service
LABEL_SELECTOR = app=quantexa-text-classifier

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

push:
	docker login
	docker tag $(IMAGE_NAME) $(DOCKER_USERNAME)/$(IMAGE_NAME):$(TAG)
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):$(TAG)

# Kubernetes commands
deploy:
	kubectl apply -f deployment.yaml
	kubectl apply -f service.yaml

delete:
	kubectl delete deployment $(DEPLOYMENT_NAME)
	kubectl delete service $(SERVICE_NAME)

# Port forwarding
port-forward:
	@echo "Setting up port forwarding for pod..."
	@POD_NAME=$$(kubectl get pods -n $(NAMESPACE) -l $(LABEL_SELECTOR) -o jsonpath="{.items[0].metadata.name}"); \
	if [ -z "$$POD_NAME" ]; then \
		echo "No pod found with label $(LABEL_SELECTOR) in namespace $(NAMESPACE)"; \
		exit 1; \
	fi; \
	echo "Setting up port forwarding for pod $$POD_NAME"; \
	nohup kubectl port-forward -n $(NAMESPACE) $$POD_NAME 9090:9090 &

# Clean up
clean:
	@echo "Stopping port forwarding..."
	@pkill -f "kubectl port-forward -n $(NAMESPACE)" || true
	@echo "Deleting Kubernetes resources..."
	kubectl delete deployment $(DEPLOYMENT_NAME)
	kubectl delete service $(SERVICE_NAME)

# Run tests
test:
	pytest

# Default target
all: build push deploy

# Help
help:
	@echo "Makefile commands:"
	@echo "  build         - Build Docker images"
	@echo "  up            - Start Docker containers"
	@echo "  down          - Stop Docker containers"
	@echo "  push          - Push Docker images to registry"
	@echo "  deploy        - Deploy to Kubernetes"
	@echo "  delete        - Delete Kubernetes deployment and service"
	@echo "  port-forward  - Set up port forwarding to the pod"
	@echo "  test          - Run tests using pytest"
	@echo "  clean         - Clean up resources"
	@echo "  all           - Build, push, and deploy the application"
	@echo "  help          - Show this help message"