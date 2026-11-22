# Local Image-to-3D Converter (CPU-Only)

A local web application that converts single 2D images (PNG/JPG) into textured 3D models (`.glb` and `.obj`) using **TripoSR** by Stability AI & Tripo AI.

Designed specifically for Windows 11 systems without dedicated Nvidia GPUs (e.g., AMD Ryzen CPU with integrated Radeon graphics). All inference runs 100% locally on the CPU with PyTorch.

---

## Architecture

- **Web UI (`app.py`)**: Built with [Gradio](https://gradio.app/) with an interactive 3D model viewer (`gr.Model3D`), live progress tracking, and an output history browser.
- **Engine Layer (`engines/`)**: Clean modular interface pattern.
  - `engines/__init__.py`: Provides `generate(image_path) -> glb_path` to the UI. The UI **never** calls engines directly.
  - `engines/local_triposr.py`: Full local pipeline: image upload → `rembg` background isolation → TripoSR neural reconstruction → marching cubes mesh extraction → `.glb` & `.obj` export.
  - `engines/cloud_api.py`: Stub engine with identical signature that raises `NotImplementedError` for future cloud API integration.
- **Configuration (`config.py`)**: Centralized settings for switching engines (`ENGINE = "local_triposr"` or `"cloud_api"`), marching cubes resolution, and server port.
- **TripoSR (`tsr/`)**: Vendored, self-contained model definitions patched with CPU-friendly `mcubes` marching cubes fallback (no Visual C++ compiler requirements on Windows).

---

## Requirements

- **OS**: Windows 10/11 (x64)
- **CPU**: AMD Ryzen or Intel x64 CPU
- **RAM**: 16 GB recommended
- **Python**: 3.10 or 3.11

---

## Quick Start

### 1. Launch via double-click
Simply double-click **`run.bat`**.
- It locates your Python installation.
- Automatically installs dependencies from `requirements.txt` on first launch.
- Starts the web server and opens your browser at `http://127.0.0.1:7860`.

### 2. Manual Terminal Setup (Alternative)
If you prefer running commands manually:

```cmd
:: 1. Install dependencies
pip install -r requirements.txt

:: 2. Launch the application
python app.py
```

Then navigate to `http://127.0.0.1:7860` in your web browser.

---

## Usage Guide

1. **Upload an Image**: Choose a PNG or JPG showing a single centered object against a plain background (toys, household items, cars, furniture work best).
2. **Remove Background**: Keep the `Remove Background (rembg)` checkbox enabled unless your image already has a transparent background.
3. **Generate**: Click **Generate 3D Model**. A progress bar and status line will report each step:
   - Background removal (`rembg`)
   - Neural network triplane prediction (`TripoSR`)
   - Surface mesh extraction (`marching cubes`)
   - `.glb` & `.obj` file export
4. **Inspect & Download**:
   - Rotate, pan, and zoom the 3D model directly in the Gradio preview.
   - Click **Download .GLB** or **Download .OBJ** to save your model.
5. **History**: Every generated model is timestamped in the `output/` directory (`output/model_YYYYMMDD_HHMMSS.glb`). The **Output History** panel lets you reload and re-download any previous creation.

---

## Switching Engines

To switch between the local engine and the cloud API stub:
Open `config.py` and set:

```python
# Use local TripoSR model:
ENGINE = "local_triposr"

# Or switch to cloud API:
ENGINE = "cloud_api"
```

The UI will automatically route calls to the selected engine through `engines.generate()`.
