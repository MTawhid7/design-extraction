# Production Deployment Guide

This guide provides instructions for deploying the Image Processing Service to a production environment on an Ubuntu 22.04 server with an NVIDIA L4 GPU.

## 1. Server Preparation

### Install Dependencies
Update the system and install essential packages, including NVIDIA drivers.

```bash
# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential git curl wget vim

# Install NVIDIA drivers (e.g., driver 535)
sudo apt install -y nvidia-driver-535
sudo reboot
```

### Install Docker and NVIDIA Container Toolkit
The service runs in Docker, which requires the NVIDIA Container Toolkit to access the GPU.

```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access in Docker
Run a test container to confirm Docker can access the GPU.

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
This should successfully display the `nvidia-smi` output.

## 2. Application Setup

### Clone the Repository
Clone the service's source code to your server.

```bash
# Create application directory
sudo mkdir -p /opt/image-processor
sudo chown $USER:$USER /opt/image-processor
cd /opt/image-processor

# Clone repository
git clone <your-repo-url> .
```

### Configure Environment
Create the `.env` file from the template and add your production credentials.

```bash
# Create .env file
cp .env.example .env

# Edit the file to add production values
nano .env```
Ensure `GEMINI_API_KEY` and a production `BASE_URL` are set.

## 3. Build Image and Download Models

Run the provided build script. This is a **one-time setup** that builds the Docker image and downloads the ML models to a persistent `./models` directory on the host.

```bash
# Make the script executable
chmod +x docker-build.sh

# Run the build process
bash docker-build.sh
```
When prompted, select `y` to download the models. This will take 5-10 minutes.

## 4. Deploy the Service

Use Docker Compose to run the service in a detached mode.

```bash
# Start the service
docker-compose up -d
```

### Verify Deployment
Check that the container is running and healthy.

```bash
# Check container status (should show 'Up' and 'healthy')
docker ps

# View service logs to ensure models loaded correctly
docker-compose logs -f

# Test the health endpoint
curl http://localhost:8001/health
```

## 5. Nginx Reverse Proxy and SSL

Set up Nginx as a reverse proxy to manage traffic and terminate SSL.

### Install and Configure Nginx

```bash
# Install Nginx
sudo apt install -y nginx

# Create a new Nginx configuration file
sudo nano /etc/nginx/sites-available/image-processor
```

Paste the following configuration, replacing `your-domain.com` with your actual domain.

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Enable the Site and Secure with SSL

```bash
# Enable the new site
sudo ln -s /etc/nginx/sites-available/image-processor /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Install Certbot for Let's Encrypt SSL
sudo apt install -y certbot python3-certbot-nginx

# Obtain and install an SSL certificate
sudo certbot --nginx -d your-domain.com

# Reload Nginx to apply changes
sudo systemctl reload nginx
```

## 6. Monitoring and Maintenance

### Monitoring
- **GPU Usage**: `watch -n 1 nvidia-smi`
- **Container Logs**: `docker-compose logs -f image-processor`
- **Nginx Logs**: `tail -f /var/log/nginx/access.log`

### Updates
To update the service with new code changes:
```bash
# Pull the latest code
git pull

# Rebuild the Docker image (models in ./models are preserved)
bash docker-build.sh

# Restart the service to apply the new image
docker-compose up -d --force-recreate
```

### Backup
Periodically back up the critical data.

```bash
# Create a compressed archive of models, config, and outputs
tar -czf backup-$(date +%Y%m%d).tar.gz /opt/image-processor/models /opt/image-processor/.env /opt/image-processor/outputs
```
