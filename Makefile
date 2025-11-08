# Makefile for the Image Processing Service

.PHONY: all build up down restart logs test shell status clean clean-full help setup download-models check-cache clean-outputs

help: ## ✨ Show this help message
	@echo "----------------------------------------------------"
	@echo " Image Processing Service - Management Commands"
	@echo "----------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- ONE-TIME SETUP ---
setup: build download-models ## 🚀 Run the full one-time setup: build image and download models

build: ## 🏗️  Build the Docker image from the Dockerfile
	@echo "--> Building Docker image..."
	@docker compose build

download-models: ## 💾 Download models to the host using your HF_TOKEN
	@echo "--> Ensuring ./models directory exists..."
	@mkdir -p models
	@echo "--> Downloading models to ./models directory (this will take several minutes)..."
	@docker run --rm --gpus all \
		--env-file .env \
		-e HF_HUB_OFFLINE=0 \
		-e HF_HOME=/models \
		--entrypoint="" \
		-v "$(pwd)/models:/models" \
		image-processor:latest \
		python3 /app/download_models.py

# --- Core Workflow ---
up: ## 🟢 Start the service in the background
	@echo "--> Starting service with docker-compose..."
	@docker compose up -d

down: ## 🔴 Stop the service
	@echo "--> Stopping service..."
	@docker compose down

restart: down up ## 🔄 Restart the service

# --- Testing & Debugging ---
test: ## 🧪 Run the test suite against the running service
	@echo "--> [1/2] Running health check..."
	@curl --fail http://localhost:8008/health || (echo "Health check failed. Is the service running? (make up)" && exit 1)
	@echo "\n\n--> [2/2] Running Python test client..."
	@venv/bin/python3 test_client.py

logs: ## 📜 View live logs from the service
	@docker compose logs -f

shell: ## 💻 Access the shell inside the running container
	@docker compose exec image-processor bash

# --- Utility & Verification ---
status: ## 📊 Show the status of the running containers
	@docker compose ps

check-cache: ## 🔍 Verify the persistent model cache on the host
	@echo "--- Verifying Persistent Model Cache in ./models ---"
	@if [ -d "./models" ] && [ "$(ls -A ./models)" ]; then \
		echo "✓ Cache directory exists and is not empty."; \
		echo "Cache Size: $$(du -sh ./models | cut -f1)"; \
		echo "--- Key Directories ---"; \
		@find ./models -maxdepth 2 -type d; \
	else \
		echo "✗ No cache directory found in ./models. Please run 'make download-models'."; \
	fi

# --- CLEAN COMMANDS ---
clean-outputs: ## 🗑️ Delete all generated images from ./outputs/
	@echo "\033[93m--> WARNING: This will permanently delete all generated images from ./outputs/\033[0m"
	@printf "Are you sure? (y/N) "; \
	read -r REPLY; \
	case "$$REPLY" in \
		[yY]*) \
			echo "--> Deleting host outputs directory..."; \
			sudo rm -rf ./outputs; \
			echo "✓ Outputs directory deleted."; \
			;; \
		*) \
			echo "Cancelled."; \
			;; \
	esac

clean: ## 🧹 Safely remove THIS PROJECT's containers and images
	@echo "--> Safely removing all containers, networks, and images for THIS project ONLY..."
	@docker compose down --rmi all -v

clean-full: ## 💥 NUCLEAR OPTION: Clean project AND DELETE the host model cache
	@echo "\033[91m--> WARNING: This will permanently delete the downloaded models from ./models/\033[0m"
	@printf "Are you sure you want to force a full re-download? (y/N) "; \
	read -r REPLY; \
	case "$$REPLY" in \
		[yY]*) \
			make clean; \
			echo "--> Deleting host model cache at ./models... (requires sudo)"; \
			sudo rm -rf ./models; \
			echo "✓ Full clean complete. Run 'bash docker-build.sh' to start fresh."; \
			;; \
		*) \
			echo "Cancelled."; \
			;; \
	esac