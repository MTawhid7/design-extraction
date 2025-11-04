# app/modules/remover/birefnet/__init__.py
"""BiRefNet background removal."""
from . import process
from .config import settings

__all__ = ["process", "settings"]