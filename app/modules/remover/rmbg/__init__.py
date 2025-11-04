# app/modules/remover/rmbg/__init__.py
"""RMBG 2.0 background removal."""
from . import process
from .config import settings

__all__ = ["process", "settings"]