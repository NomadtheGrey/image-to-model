"""Local TripoSR engine for image-to-3D model generation on CPU."""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image

import config
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

# Global cached model and rembg session for reuse across requests
_model: Optional[TSR] = None
_rembg_session = None


def get_rembg_session():
    """Lazily initialize and cache rembg session."""
    global _rembg_session
    if _rembg_session is None:
        import rembg
        _rembg_session = rembg.new_session()
    return _rembg_session


def get_model() -> TSR:
    """Lazily load and cache TripoSR model on CPU."""
    global _model
    if _model is None:
        cfg = config.TRIPOSR_CONFIG
        device = cfg.get("device", "cpu")
        print(f"[local_triposr] Loading TripoSR model '{cfg['model_id']}' onto {device}...")
        
        _model = TSR.from_pretrained(
            cfg["model_id"],
            config_name=cfg["config_name"],
            weight_name=cfg["weight_name"],
        )
        chunk_size = cfg.get("chunk_size", 8192)
        _model.renderer.set_chunk_size(chunk_size)
        _model.to(device)
        print(f"[local_triposr] TripoSR model loaded successfully.")
    return _model


def fill_background(image: Image.Image) -> Image.Image:
    """Composite RGBA image onto a 50% gray background."""
    image_arr = np.array(image).astype(np.float32) / 255.0
    if image_arr.shape[-1] == 4:
        rgb = image_arr[:, :, :3]
        alpha = image_arr[:, :, 3:4]
        composite = rgb * alpha + (1.0 - alpha) * 0.5
        return Image.fromarray((composite * 255.0).astype(np.uint8))
    return image.convert("RGB")


def generate(
    image_path: str,
    remove_bg: bool = True,
    foreground_ratio: float = 0.85,
    mc_resolution: int = 256,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    export_obj: bool = True,
) -> str:
    """Convert an image file to a 3D model (.glb and .obj) using local TripoSR.

    Args:
        image_path: Path to the input image (PNG/JPG).
        remove_bg: Whether to run rembg background removal.
        foreground_ratio: Ratio of the foreground subject relative to the canvas.
        mc_resolution: Marching cubes resolution for mesh extraction.
        progress_callback: Optional callable (progress: float, message: str).
        export_obj: Whether to also save an OBJ file alongside the GLB.

    Returns:
        str: Absolute path to the generated .glb file.
    """
    def update_progress(val: float, msg: str):
        print(f"[local_triposr] [{int(val * 100)}%] {msg}")
        if progress_callback:
            progress_callback(val, msg)

    start_time = time.time()
    update_progress(0.05, "Opening input image...")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGBA")

    # Step 1: Background removal & preprocessing
    if remove_bg:
        update_progress(0.15, "Removing background with rembg...")
        session = get_rembg_session()
        pil_img = remove_background(pil_img, session)
        pil_img = resize_foreground(pil_img, foreground_ratio)
        pil_img = fill_background(pil_img)
    else:
        update_progress(0.15, "Preprocessing image (keeping background)...")
        if pil_img.mode == "RGBA":
            pil_img = fill_background(pil_img)
        else:
            pil_img = pil_img.convert("RGB")

    # Step 2: Load model onto CPU
    update_progress(0.30, "Preparing TripoSR model on CPU...")
    model = get_model()
    device = config.TRIPOSR_CONFIG.get("device", "cpu")

    # Step 3: Run feed-forward inference
    update_progress(0.45, "Running TripoSR neural network inference...")
    with torch.no_grad():
        scene_codes = model(pil_img, device=device)

    # Step 4: Marching cubes mesh extraction
    update_progress(0.70, f"Extracting 3D surface mesh (resolution: {mc_resolution})...")
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
        mesh = meshes[0]

    update_progress(0.85, "Orienting mesh for 3D viewer...")
    mesh = to_gradio_3d_orientation(mesh)

    # Step 5: Export to output directory with timestamp
    update_progress(0.92, "Exporting 3D model files...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_filename = f"model_{timestamp}.glb"
    glb_path = str(output_dir / glb_filename)
    mesh.export(glb_path)

    if export_obj:
        obj_filename = f"model_{timestamp}.obj"
        obj_path = str(output_dir / obj_filename)
        mesh.export(obj_path)

    elapsed = time.time() - start_time
    update_progress(1.0, f"Model generated in {elapsed:.1f}s: {glb_filename}")

    return glb_path
