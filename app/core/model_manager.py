"""
Centralized model manager for all ML models.
Optimized for L4 GPU with 24GB VRAM.
Updated for google-genai SDK (2025).
"""
import torch
import structlog
import asyncio
from typing import Optional
from google import genai
from transformers import AutoModelForImageSegmentation, AutoProcessor
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os

from app.config import settings
# --- NEW: Import the IS-Net model architecture ---
from app.modules.remover.isnet.isnet_model import ISNetDIS

log = structlog.get_logger(__name__)
CACHE_DIR = os.getenv("HF_HOME", "/models")

class ModelManager:
    """Manages all ML models with efficient GPU memory usage."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False
        self.gemini_client: Optional[genai.Client] = None

        # --- REMOVED: rmbg and birefnet models ---
        # self.rmbg_model = None
        # self.rmbg_processor = None
        # self.birefnet_model = None

        # --- ADDED: isnet model ---
        self.isnet_model = None

        self.realesrgan_model = None
        log.info("ModelManager created", device=str(self.device))

    async def initialize(self):
        if self.is_initialized: return
        log.info("Starting model initialization...")
        await self._init_gemini()

        # --- MODIFIED: Simplified model loading ---
        await asyncio.gather(
            self._init_isnet(),
            self._init_realesrgan(),
        )

        self.is_initialized = True
        log.info("All models initialized successfully")
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            log.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    async def _init_gemini(self):
        log.info("Initializing Gemini client...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=api_key)
        log.info("Gemini Client initialized with google-genai SDK")

    # --- REMOVED: _init_rmbg method ---

    # --- REMOVED: _init_birefnet method ---

    # --- ADDED: _init_isnet method ---
    async def _init_isnet(self):
        """Initialize the IS-Net background removal model."""
        log.info("Loading IS-Net model...")
        model_path = os.path.join(CACHE_DIR, "isnet-general-use.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"IS-Net model not found at {model_path}. Please add it to the ./models directory.")

        self.isnet_model = ISNetDIS()
        self.isnet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.isnet_model.to(self.device)
        if self.device.type == "cuda":
            self.isnet_model.half()
        self.isnet_model.eval()
        log.info("IS-Net model loaded successfully.")

    async def _init_realesrgan(self):
        """Initialize Real-ESRGAN upscaling model from the local cache."""
        log.info("Loading Real-ESRGAN model...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join(CACHE_DIR, "xinntao_Real-ESRGAN", "RealESRGAN_x4plus.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RealESRGAN model not found at {model_path}. Please run 'make download-models'.")
        use_fp16 = self.device.type == "cuda"
        self.realesrgan_model = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            half=use_fp16,
            gpu_id=0 if use_fp16 else None,
        )
        log.info("Real-ESRGAN model loaded from cache.")

    async def cleanup(self):
        log.info("Cleaning up ModelManager...")
        # --- REMOVED: rmbg and birefnet cleanup ---
        # if self.rmbg_model: self.rmbg_model.cpu()
        # if self.birefnet_model: self.birefnet_model.cpu()

        # --- ADDED: isnet cleanup ---
        if self.isnet_model: self.isnet_model.cpu()

        if self.realesrgan_model: self.realesrgan_model = None
        if self.gemini_client:
            if hasattr(self.gemini_client, 'aio'):
                await self.gemini_client.aio.aclose()
        if self.device.type == "cuda": torch.cuda.empty_cache()
        self.is_initialized = False
        log.info("ModelManager cleanup complete")