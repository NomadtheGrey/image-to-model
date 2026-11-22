"""Engine interface module.

Provides a unified interface for 3D model generation engines.
The active engine is determined by config.ENGINE ('local_triposr' or 'cloud_api').
The UI should always call engines.generate(...) rather than calling an engine directly.
"""
from typing import Callable, Optional
import config


def get_engine():
    """Retrieve the engine module configured in config.py."""
    engine_name = getattr(config, "ENGINE", "local_triposr").lower()
    if engine_name == "local_triposr":
        from . import local_triposr
        return local_triposr
    elif engine_name == "cloud_api":
        from . import cloud_api
        return cloud_api
    else:
        raise ValueError(f"Unknown engine '{engine_name}' configured in config.py")


def generate(
    image_path: str,
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
    mc_resolution: int = 256,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    export_obj: bool = True,
) -> str:
    """Generate a 3D model (.glb) from an image using the active engine.

    Args:
        image_path: Path to input image file.
        remove_bg: Whether background removal should be performed.
        foreground_ratio: Ratio of the object within the bounding canvas.
        mc_resolution: Marching cubes grid resolution.
        progress_callback: Callback receiving (fraction_complete, status_text).
        export_obj: Whether to also save an OBJ file.

    Returns:
        str: Path to the generated .glb file.
    """
    engine = get_engine()
    return engine.generate(
        image_path=image_path,
        remove_bg=remove_bg,
        foreground_ratio=foreground_ratio,
        mc_resolution=mc_resolution,
        progress_callback=progress_callback,
        export_obj=export_obj,
    )


__all__ = ["generate", "get_engine"]
