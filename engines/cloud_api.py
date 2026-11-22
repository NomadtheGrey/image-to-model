"""Stub cloud API engine for image-to-3D model generation."""
from typing import Callable, Optional


def generate(
    image_path: str,
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
    mc_resolution: int = 256,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    export_obj: bool = True,
) -> str:
    """Stub engine for cloud-based image-to-3D API.

    Raises:
        NotImplementedError: Cloud API is a stub for future expansion.
    """
    raise NotImplementedError(
        "Cloud API engine is not implemented yet. "
        "Please set ENGINE = 'local_triposr' in config.py."
    )
