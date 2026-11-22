"""Configuration for the Image-to-3D converter application."""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Engine selection: "local_triposr" or "cloud_api"
# The UI loads whichever engine is specified here through engines/__init__.py
ENGINE = "local_triposr"

# Local TripoSR Engine Configuration
TRIPOSR_CONFIG = {
    "model_id": "stabilityai/TripoSR",
    "config_name": "config.yaml",
    "weight_name": "model.ckpt",
    "device": "cpu",  # CPU-only execution for AMD Ryzen / non-Nvidia hardware
    "chunk_size": 8192,
    "default_mc_resolution": 256,
    "default_foreground_ratio": 0.85,
    "default_remove_bg": True,
}

# Server configuration
SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
