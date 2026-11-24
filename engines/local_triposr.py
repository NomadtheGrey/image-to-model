"""Local TripoSR engine for image-to-3D model generation on CPU."""
import os
import threading
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

# A progress reporter receives (fraction_complete, status_text). The engine
# reports every stage through it — including the first-run model downloads,
# which the underlying libraries perform silently. A stage that reports
# nothing looks exactly like a hang, so nothing in generate() is allowed
# to be silent.
Reporter = Callable[[float, str], None]


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under a directory (0 when it does not exist)."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _run_while_reporting(work, watch_dir: Path, on_poll) -> None:
    """Run work() in a thread, calling on_poll(bytes_so_far) as files grow.

    The model files are fetched by library calls that give the UI no
    progress at all. Polling the folder they land in is what turns a
    silent multi-minute download into a number the user can watch.
    on_poll is only called when the byte count changes.
    """
    errors = []

    def runner():
        try:
            work()
        except Exception as exc:  # surfaced on the caller's thread below
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    last_size = -1
    while thread.is_alive():
        thread.join(timeout=1.0)
        size = _dir_size_bytes(watch_dir)
        if size != last_size:
            last_size = size
            on_poll(size)
    if errors:
        raise errors[0]


def _repo_cache_dir(model_id: str) -> Path:
    """Hugging Face hub cache folder for a repo (where downloads land)."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + model_id.replace("/", "--"))


def _repo_total_bytes(model_id: str) -> Optional[int]:
    """Total download size of a HF repo, or None when it cannot be asked."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        sizes = [s.size for s in info.siblings if s.size]
        if sizes:
            return sum(sizes)
    except Exception:
        return None
    return None


def ensure_triposr_files(report: Reporter):
    """Make sure the TripoSR weights are on disk, narrating any download."""
    from huggingface_hub import hf_hub_download

    cfg = config.TRIPOSR_CONFIG
    model_id = cfg["model_id"]
    filenames = [cfg["config_name"], cfg["weight_name"]]
    try:
        for name in filenames:
            hf_hub_download(repo_id=model_id, filename=name, local_files_only=True)
        return  # Already cached — later runs land here and skip the download.
    except Exception:
        pass  # Not fully cached: fall through and download with narration.

    total = _repo_total_bytes(model_id)
    total_text = f" of about {total / 1e6:.0f} MB" if total else ""

    def on_poll(have: int):
        frac = 0.18 + 0.10 * min(have / total, 1.0) if total else 0.22
        report(
            frac,
            "Downloading the TripoSR model from Hugging Face (one-time — "
            f"later runs skip this): {have / 1e6:.0f} MB{total_text} so far. "
            "This is a network wait; the speed is your connection's.",
        )

    def work():
        for name in filenames:
            hf_hub_download(repo_id=model_id, filename=name)

    _run_while_reporting(work, _repo_cache_dir(model_id), on_poll)


def get_rembg_session(report: Reporter):
    """Lazily initialize and cache rembg session, narrating its download.

    rembg fetches its u2net model (~170 MB) the first time a session is
    created and says nothing while it does — on a slow connection that
    looks exactly like the app hanging during 'Removing background...'.
    """
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session

    import rembg

    u2net_home = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    if (u2net_home / "u2net.onnx").exists():
        _rembg_session = rembg.new_session()
        return _rembg_session

    holder = {}

    def on_poll(have: int):
        # u2net.onnx is ~176 MB; the fraction is a guide, the MB count is fact.
        frac = 0.04 + 0.08 * min(have / 176e6, 1.0)
        report(
            frac,
            "Downloading the background-removal model (one-time, about "
            f"170 MB — later runs skip this): {have / 1e6:.0f} MB so far. "
            "This is a network wait; the speed is your connection's.",
        )

    def work():
        holder["session"] = rembg.new_session()

    _run_while_reporting(work, u2net_home, on_poll)
    _rembg_session = holder["session"]
    return _rembg_session


def get_model(report: Reporter) -> TSR:
    """Lazily load and cache TripoSR model on CPU, narrating each phase."""
    global _model
    if _model is not None:
        return _model

    cfg = config.TRIPOSR_CONFIG
    device = cfg.get("device", "cpu")
    ensure_triposr_files(report)
    report(
        0.30,
        "Loading the model into memory — this takes a moment even with "
        "nothing to download...",
    )
    print(f"[local_triposr] Loading TripoSR model '{cfg['model_id']}' onto {device}...")
    _model = TSR.from_pretrained(
        cfg["model_id"],
        config_name=cfg["config_name"],
        weight_name=cfg["weight_name"],
    )
    chunk_size = cfg.get("chunk_size", 8192)
    _model.renderer.set_chunk_size(chunk_size)
    _model.to(device)
    print("[local_triposr] TripoSR model loaded successfully.")
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
    update_progress(0.02, "Opening your image...")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    pil_img = Image.open(image_path).convert("RGBA")

    # Step 1: Background removal & preprocessing. Creating the rembg
    # session can itself download its model — get_rembg_session narrates
    # that instead of letting it hide inside this stage.
    if remove_bg:
        session = get_rembg_session(update_progress)
        update_progress(0.15, "Removing the background...")
        pil_img = remove_background(pil_img, session)
        pil_img = resize_foreground(pil_img, foreground_ratio)
        pil_img = fill_background(pil_img)
    else:
        update_progress(0.15, "Preprocessing image (keeping background)...")
        if pil_img.mode == "RGBA":
            pil_img = fill_background(pil_img)
        else:
            pil_img = pil_img.convert("RGB")

    # Step 2: Load model onto CPU. get_model narrates the one-time
    # weights download separately from the load itself — the two waits
    # feel identical from the user's chair and must not share one label.
    model = get_model(update_progress)
    device = config.TRIPOSR_CONFIG.get("device", "cpu")
    update_progress(0.38, "Model loaded and ready.")

    # Step 3: Run feed-forward inference
    update_progress(0.45, "Studying the image and imagining the 3D shape...")
    with torch.no_grad():
        scene_codes = model(pil_img, device=device)

    # Step 4: Marching cubes mesh extraction — the long silent compute
    # stage, so the label carries the expectation with it.
    update_progress(
        0.70,
        f"Carving the mesh at resolution {mc_resolution} — the slow step "
        "(about a minute at 128; 256 takes much longer)...",
    )
    with torch.no_grad():
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
        mesh = meshes[0]

    update_progress(0.85, "Orienting the model for the viewer...")
    mesh = to_gradio_3d_orientation(mesh)

    # Step 5: Export to output directory with timestamp
    update_progress(0.92, "Saving the model files...")
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
