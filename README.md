# High-Performance Image Processing Service

This service provides a GPU-accelerated, parallel image processing pipeline. It is optimized for intermittent use on servers like the NVIDIA L4, featuring a fast start/stop workflow.

The core strategy is to use a small Docker image and a **persistent model cache** stored on the host machine. This means models are downloaded only once, enabling the container to start and become fully operational in under 30 seconds.

The service now uses a streamlined pipeline featuring **Google Gemini** for design extraction, **IS-Net** for high-fidelity background removal, and **Real-ESRGAN** for 4x upscaling.

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
```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Configure the Environment
Copy the example `.env` file and add your Google Gemini API key.

```bash
cp .env.example .env
nano .env
```
- **`GEMINI_API_KEY`**: (Required) Your Google Gemini API key.
- **`BASE_URL`**: The public-facing URL for your service (e.g., `http://localhost:8008` for local testing).

### 2. Place the IS-Net Model Manually (Critical Step)
Due to its research origins, the IS-Net model weights are not downloaded by the script. You must place them in the `models` directory yourself.

1.  **Create the directory:**
    ```bash
    mkdir -p models
    ```
2.  **Copy your model file** into this directory. The file **must** be named exactly **`isnet-general-use.pth`**.

The final path must be: `./models/isnet-general-use.pth`

### 3. One-Time Setup Command
This single command builds the Docker image and runs the script to download the remaining required models (Real-ESRGAN) to the `./models` directory.

```bash
make setup
```

### 4. Start the Service
This command starts the container in the background. The application will then take 10-30 seconds to load the models from the cache into GPU memory.

```bash
make up
```

### 5. Verify and Test
- **Check the Logs**: View the service logs to confirm a clean startup.
  ```bash
  make logs
  ```
  Look for `All models initialized successfully`.

- **Verify the Cache**: Run the cache check to confirm the models are correctly stored on the host.
  ```bash
  make check-cache
  ```
  You should see both the `isnet-general-use.pth` file and the `xinntao_Real-ESRGAN` directory listed.

- **Run the Test Suite**: Execute the automated test script.
  ```bash
  make test
  ```
  The response should show a successful health check and a completed image processing request with the new output format.

## 🔧 Common Commands

Use these `make` commands to manage the service.

| Command | Description |
| :--- | :--- |
| `make help` | ✨ Show all available commands. |
| `make setup` | 🚀 **(Run once)** Build image & download public models. |
| `make up` | 🟢 Start the service in the background. |
| `make down` | 🔴 Stop the service. |
| `make restart` | 🔄 Restart the service. |
| `make logs` | 📜 View live logs from the service. |
| `make test` | 🧪 Run the health check and test client against the service. |
| `make shell` | 💻 Access a `bash` shell inside the running container. |
| `make status` | 📊 Show the status of the running containers. |
| `make check-cache` | 🔍 Verify the contents of the persistent `./models` cache. |
| `make clean-outputs` | 🗑️ Delete all generated images from `./outputs/`. |
| `make clean` | 🧹 Safely stop and remove **this project's** containers and images. |
| `make clean-full`| 💥 **NUCLEAR OPTION:** Clean project AND DELETE all models from `./models/`. |


## API Reference

### `POST /process`
Processes front and back images through the full Gemini -> IS-Net -> Real-ESRGAN pipeline.

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
  "front_output": "http://localhost:8008/design_123..._front_isnet.png",
  "back_output": "http://localhost:8008/design_123..._back_isnet.png",
  "processing_time_seconds": 12.5
}
```

### `GET /health`
Checks the service and model loading status.

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## Performance & Optimization

- **GPU**: NVIDIA L4 24GB
- **Startup Time**: ~25 seconds (container start + model load)
- **Processing Time**: 7-14 seconds per request
- **Peak VRAM Usage**: ~1.1GB **per worker** (e.g., `WORKERS=2` may use ~2.2GB)

## 💾 Backup Strategy

The most critical asset is the downloaded model cache.

- **What to back up**:
  - `./models/` (Essential model files, including `isnet-general-use.pth`)
  - `.env` (API Keys and configuration)
  - `./outputs/` (Generated images, if needed)

- **Backup Command**:
  ```bash
  tar -czf backup-$(date +%Y%m%d).tar.gz models/ .env outputs/
  ```

## 🐛 Troubleshooting

- **`Bind for ... port is already allocated` on `make up`**: The host port (e.g., `8008`) is in use by another process.
  - **Solution**: Check what is using the port with `sudo lsof -i :8008`. If needed, change the host-side port in `docker-compose.yml`.

- **`FATAL ERROR: One or more model files were not found` on `make up`**: The model cache is missing required files.
  - **Solution**:
    1.  Verify that `isnet-general-use.pth` exists directly inside the `./models/` directory.
    2.  Run `make download-models` to ensure the Real-ESRGAN files are present.
    3.  Use `make check-cache` to see what is missing.

- **`ModuleNotFoundError` on `make test`**: The test client's dependencies are not installed on the host.
  - **Solution**: Follow the "For Local Testing" instructions in the Prerequisites section to set up and activate a virtual environment.