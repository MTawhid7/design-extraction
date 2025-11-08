# app/core/process_lock.py
from filelock import FileLock
from pathlib import Path

# Create a directory for lock files to keep things clean.
lock_dir = Path("/tmp/app_locks")
lock_dir.mkdir(exist_ok=True)

# A dedicated lock for the IS-Net model.
# Only one process can run IS-Net inference at a time.
isnet_lock = FileLock(lock_dir / "isnet_model.lock", timeout=120) # 2 minute timeout

# A dedicated lock for the Real-ESRGAN model.
# Only one process can run upscaling at a time.
realesrgan_lock = FileLock(lock_dir / "realesrgan_model.lock", timeout=120)