# High-Performance Image Processing Service

This service provides a GPU-accelerated, parallel image processing pipeline. It is optimized for intermittent use on servers like the NVIDIA L4, featuring a fast start/stop workflow.

The core strategy is to use a small Docker image and a **persistent model cache** stored on the host machine. This means models are downloaded only once, enabling the container to start and become fully operational in under 30 seconds.

## Management via Makefile

All project operations (building, running, testing, cleaning) are managed through a `Makefile`. This provides a simple, consistent, and safe command interface.

To see all available commands, simply run:
```bash
make
```

## 🚀 Quick Start (5-Minute Setup)

### Prerequisites
- Docker and Docker Compose (V2)
- `make`
- NVIDIA Container Toolkit (for GPU support)
- A working `nvidia-smi` command on the host

### For Local Testing (`make test`)
The test client runs on the host. You must install its dependencies in a local virtual environment.
# Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

# Install dependencies
```
pip install -r requirements.txt
```

### 1. Configure the Environment
This is the most critical step. You must provide your API keys for the service to function.

1.  **Copy the example `.env` file:**
    ```bash
    cp .env.example .env
    ```
2.  **Edit the `.env` file:**
    ```bash
    nano .env
    ```
3.  **Add your keys and URL:**
    - `GEMINI_API_KEY`: Your Google Gemini API key.
    - `HF_TOKEN`: Your Hugging Face token. This is **required** to download the gated RMBG 2.0 model. Ensure you have accepted the license for `briaai/RMBG-2.0` on the Hugging Face website.
    - `BASE_URL`: The public-facing URL for your service (e.g., `http://localhost:8008` for local testing).

### 2. One-Time Setup (Build and Download)
This single command builds the Docker image and runs a dedicated script to download all required models to the `./models` directory on your host. This will take 5-10 minutes.

```bash
make setup
```

### 3. Start the Service
This command starts the container in the background. The application will then take 10-30 seconds to load the models from the cache into GPU memory.

```bash
make up
```

### 4. Verify and Test
- **Check the Logs**: View the service logs to confirm a clean startup.
  ```bash
  make logs
  ```
  Look for `All models initialized successfully`.

- **Verify the Cache**: Run the cache check to confirm the models are correctly stored on the host.
  ```bash
  make check-cache
  ```
  You should see a large cache size (e.g., `2.2G`) and a list of the key model directories.

- **Run the Test Suite**: Execute the automated test script.
  ```bash
  make test
  ```
  The response should show a successful health check and a completed image processing request.

## 🔧 Common Commands

Use these `make` commands to manage the service.

| Command | Description |
| :--- | :--- |
| `make help` | ✨ Show all available commands. |
| `make setup` | 🚀 **(Run once)** Build the image and download all models. |
| `make up` | 🟢 Start the service in the background. |
| `make down` | 🔴 Stop the service. |
| `make restart` | 🔄 Restart the service. |
| `make logs` | 📜 View live logs from the service. |
| `make test` | 🧪 Run the health check and test client. |
| `make shell` | 💻 Access a `bash` shell inside the running container for debugging. |
| `make status` | 📊 Show the status of the running containers. |
| `make check-cache` | 🔍 Verify the status and contents of the persistent model cache. |
| `make clean` | 🧹 Safely stop and remove **this project's** containers and images. |
| `make clean-full`| 🗑️ **(DANGEROUS)** Clean the project AND delete the downloaded models from `./models/`. |

## API Reference

### `POST /process`
Processes front and back images through the full pipeline.

**Request Body**:
```json
{
  "id": 123,
  "output": {
    "front": "https://example.com/front.png",
    "back": "https://example.com/back.png"
  }
}
```
**Response (200 OK)** (assuming `BASE_URL=http://localhost:8008`):
```json
{
  "id": 123,
  "front_rmbg": "http://localhost:8008/design_123..._front_rmbg.png",
  "front_birefnet": "http://localhost:8008/design_123..._front_birefnet.png",
  "back_rmbg": "http://localhost:8008/design_123..._back_rmbg.png",
  "back_birefnet": "http://localhost:8008/design_123..._back_birefnet.png",
  "processing_time_seconds": 14.7
}
```

### `GET /health`
Checks the service and model loading status.

## Performance & Optimization

- **GPU**: NVIDIA L4 24GB
- **Startup Time**: ~30 seconds (container start + model load)
- **Processing Time**: 8-15 seconds per request
- **Peak VRAM Usage**: ~1.3GB

## 💾 Backup Strategy

The most critical asset is the downloaded model cache.

- **What to back up**:
  - `./models/` (Essential model files)
  - `.env` (API Keys and configuration)
  - `./outputs/` (Generated images, if needed)

- **Backup Command**:
  ```bash
  tar -czf backup-$(date +%Y%m%d).tar.gz models/ .env outputs/
  ```

## 🐛 Troubleshooting

- **`Bind for ... port is already allocated` on `make up`**: The host port (e.g., `8008`) is in use by another process.
  - **Solution**: Check what is using the port with `sudo lsof -i :8008`. If needed, change the host-side port in `docker-compose.yml` to an unused one (e.g., `8009:8001`).

- **`FATAL ERROR: Model files not found` on `make up`**: This means the model cache is empty.
  - **Solution**: Run `make download-models` to populate the cache, then try `make up` again. Use `make check-cache` to verify.

- **`ModuleNotFoundError` on `make test`**: The test client's dependencies are not installed on the host.
  - **Solution**: Follow the "For Local Testing" instructions in the Prerequisites section to set up a virtual environment.

- **`401 Client Error` on `make download-models`**: This is an authentication error with Hugging Face.
  - **Solution**: Ensure your `HF_TOKEN` in `.env` is correct and that you have accepted the license for `briaai/RMBG-2.0` on the Hugging Face website.

- **`CUDA out of memory`**: The GPU has run out of VRAM.
  - **Solution**: Run `make restart` to clear the GPU memory. Ensure no other processes are using the GPU.
