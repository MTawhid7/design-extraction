"""
Centralized model manager for all ML models.
Optimized for L4 GPU with 24GB VRAM.
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
# --- IMPORT THE GEMINI CONFIGS DIRECTLY FOR INITIALIZATION ---
from app.modules.extractor.config import GENERATION_CONFIG, SAFETY_SETTINGS

log = structlog.get_logger(__name__)
CACHE_DIR = os.getenv("HF_HOME", "/models")

class ModelManager:
    """Manages all ML models with efficient GPU memory usage."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False
        # --- The manager now holds the high-level GenerativeModel object ---
        self.gemini_model: Optional[genai.GenerativeModel] = None
        self.rmbg_model = None
        self.rmbg_processor = None
        self.birefnet_model = None
        self.realesrgan_model = None
        log.info("ModelManager created", device=str(self.device))

    async def initialize(self):
        if self.is_initialized: return
        log.info("Starting model initialization...")
        await self._init_gemini()
        await asyncio.gather(
            self._init_rmbg(),
            self._init_birefnet(),
            self._init_realesrgan(),
        )
        self.is_initialized = True
        log.info("All models initialized successfully")
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            log.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    async def _init_gemini(self):
        log.info("Initializing Gemini model...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY environment variable not set")

        # --- THIS IS THE CORRECTED, MODERN INITIALIZATION ---
        # The GenerativeModel helper is the standard, robust way to interact with the API.
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL_NAME,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS
        )
        log.info("Gemini GenerativeModel initialized")

    async def _init_rmbg(self):
        log.info("Loading RMBG 2.0 model from local path...")
        local_model_path = os.path.join(CACHE_DIR, "models--briaai--RMBG-2.0", "snapshots", "a6a8895f89cf3150d2046e004766d2b93712c337")
        self.rmbg_model = AutoModelForImageSegmentation.from_pretrained(
            local_model_path, trust_remote_code=True, local_files_only=True
        )
        self.rmbg_processor = AutoProcessor.from_pretrained(
            local_model_path, trust_remote_code=True, local_files_only=True
        )
        self.rmbg_model.to(self.device)
        if self.device.type == "cuda": self.rmbg_model.half()
        self.rmbg_model.eval()
        log.info("RMBG 2.0 model loaded from explicit local path.")

    async def _init_birefnet(self):
        log.info("Loading BiRefNet-HR model from local path...")
        local_model_path = os.path.join(CACHE_DIR, "models--ZhengPeng7--BiRefNet_HR-matting", "snapshots", "4548a3861993fb5a6f174dd2b5b52b9dbc226769")
        self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            local_model_path, trust_remote_code=True, local_files_only=True
        )
        self.birefnet_model.to(self.device)
        if self.device.type == "cuda": self.birefnet_model.half()
        self.birefnet_model.eval()
        log.info("BiRefNet-HR model loaded from explicit local path.")

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
        if self.rmbg_model: self.rmbg_model.cpu()
        if self.birefnet_model: self.birefnet_model.cpu()
        if self.realesrgan_model: self.realesrgan_model = None
        if self.device.type == "cuda": torch.cuda.empty_cache()
        self.is_initialized = False
        log.info("ModelManager cleanup complete")