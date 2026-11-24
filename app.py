"""Gradio web application for Image-to-3D model conversion (CPU-only)."""
import os
import glob
import queue
import threading
import time
from pathlib import Path
from typing import List

import gradio as gr
import config
import engines


def get_history_files() -> List[str]:
    """Scan the output folder and return filenames sorted by newest first."""
    output_dir = Path(config.OUTPUT_DIR)
    glb_files = sorted(output_dir.glob("*.glb"), key=os.path.getmtime, reverse=True)
    return [f.name for f in glb_files]


def on_generate(
    image_path: str,
    remove_bg: bool,
    foreground_ratio: float,
    mc_resolution: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Handle generation button click, streaming live status to the UI.

    This is a generator on purpose: the engine runs in a worker thread and
    posts every stage change to a queue, and each post becomes a status
    line the user can read while the run continues. When a stage runs long
    without posting (the mesh carve is one call), a heartbeat re-posts the
    current stage with an elapsed counter — a status that visibly ticks is
    how the user tells "working" from "hung" without opening Task Manager.
    """
    if not image_path:
        yield None, None, None, "Please upload an image before clicking Generate.", gr.update()
        return

    events: "queue.Queue[tuple[float, str]]" = queue.Queue()
    result = {}

    def run_engine():
        try:
            # UI strictly calls engines.generate — never the engine directly.
            result["glb_path"] = engines.generate(
                image_path=image_path,
                remove_bg=remove_bg,
                foreground_ratio=foreground_ratio,
                mc_resolution=int(mc_resolution),
                progress_callback=lambda val, msg: events.put((val, msg)),
                export_obj=True,
            )
        except Exception as exc:  # reported on the handler thread below
            result["error"] = exc

    worker = threading.Thread(target=run_engine, daemon=True)
    started = time.time()
    worker.start()

    last_msg = "Starting generation..."
    while worker.is_alive():
        try:
            val, msg = events.get(timeout=2.0)
            last_msg = msg
            progress(val, desc=msg)
            yield gr.update(), gr.update(), gr.update(), msg, gr.update()
        except queue.Empty:
            elapsed = int(time.time() - started)
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                f"{last_msg} — {elapsed}s elapsed",
                gr.update(),
            )
    worker.join()

    if "error" in result:
        exc = result["error"]
        print(f"[app] Error: {exc}")
        yield None, None, None, f"Generation failed: {exc}", gr.update()
        return

    glb_path = result["glb_path"]
    obj_path = glb_path.replace(".glb", ".obj")
    if not os.path.exists(obj_path):
        obj_path = None

    filename = os.path.basename(glb_path)
    elapsed = int(time.time() - started)
    status_msg = f"Done in {elapsed}s! Generated 3D model saved to output/{filename}"

    # Refresh history list
    updated_history = get_history_files()
    yield (
        glb_path,        # 3D preview
        glb_path,        # Download GLB
        obj_path,        # Download OBJ
        status_msg,      # Status text
        gr.update(choices=updated_history, value=filename),
    )


def on_select_history(selected_name: str):
    """Load an existing model from the output folder into the viewer and downloaders."""
    if not selected_name:
        return None, None, None, "Select a model from the history list."

    glb_path = str(Path(config.OUTPUT_DIR) / selected_name)
    if not os.path.exists(glb_path):
        return None, None, None, f"File not found: {selected_name}"

    obj_path = glb_path.replace(".glb", ".obj")
    if not os.path.exists(obj_path):
        obj_path = None

    return (
        glb_path,
        glb_path,
        obj_path,
        f"Previewing saved model: {selected_name}",
    )


def on_refresh_history():
    """Refresh the history dropdown choices."""
    history_files = get_history_files()
    latest = history_files[0] if history_files else None
    return gr.update(choices=history_files, value=latest)


def create_ui():
    """Build the Gradio UI blocks."""
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto; }
    #status-box { font-weight: 500; font-size: 1.05rem; }
    """

    with gr.Blocks(title="Local Image-to-3D Converter") as demo:
        gr.Markdown(
            """
            # Local Image-to-3D Converter
            Convert single-object photos or illustrations into 3D models (`.glb` & `.obj`) using **TripoSR**.
            *Running 100% locally on CPU (AMD Ryzen / Integrated Radeon).*
            """
        )

        with gr.Row():
            # Left column: Controls and Image upload
            with gr.Column(scale=5):
                input_image = gr.Image(
                    label="Input Image (single object on plain background works best)",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                
                with gr.Group():
                    remove_bg_toggle = gr.Checkbox(
                        label="Remove Background (rembg)",
                        value=config.TRIPOSR_CONFIG.get("default_remove_bg", True),
                        info="Automatically isolates the main object from the background.",
                    )
                    with gr.Accordion("Advanced Settings", open=False):
                        foreground_ratio = gr.Slider(
                            label="Foreground Subject Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=config.TRIPOSR_CONFIG.get("default_foreground_ratio", 0.85),
                            step=0.05,
                            info="Bounding box scale of the object on the canvas.",
                        )
                        mc_resolution = gr.Slider(
                            label="Mesh Resolution (Marching Cubes)",
                            minimum=64,
                            maximum=256,
                            value=config.TRIPOSR_CONFIG.get("default_mc_resolution", 256),
                            step=32,
                            info="256 is highest quality. 128 or 160 is faster on CPU.",
                        )

                generate_btn = gr.Button("Generate 3D Model", variant="primary", size="lg")
                status_box = gr.Markdown(
                    "Ready. Upload an image and click Generate. The status "
                    "line here narrates every stage as it happens — on the "
                    "very first run that includes one-time model downloads "
                    "(about 1.8 GB total); later runs skip those.",
                    elem_id="status-box",
                )

            # Right column: 3D Preview and Downloads
            with gr.Column(scale=6):
                model_preview = gr.Model3D(
                    label="3D Model Preview",
                    interactive=True,
                )
                with gr.Row():
                    download_glb = gr.File(label="Download .GLB Model", interactive=False)
                    download_obj = gr.File(label="Download .OBJ Model", interactive=False)

        # Bottom section: Generation History
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Output History")
                with gr.Row():
                    history_dropdown = gr.Dropdown(
                        label="Saved Models in output/",
                        choices=get_history_files(),
                        value=get_history_files()[0] if get_history_files() else None,
                        scale=4,
                    )
                    refresh_history_btn = gr.Button("Refresh History", scale=1)
                load_history_btn = gr.Button("Load Selected Model", variant="secondary")

        # Event handlers
        generate_btn.click(
            fn=on_generate,
            inputs=[input_image, remove_bg_toggle, foreground_ratio, mc_resolution],
            outputs=[model_preview, download_glb, download_obj, status_box, history_dropdown],
        )

        load_history_btn.click(
            fn=on_select_history,
            inputs=[history_dropdown],
            outputs=[model_preview, download_glb, download_obj, status_box],
        )

        history_dropdown.change(
            fn=on_select_history,
            inputs=[history_dropdown],
            outputs=[model_preview, download_glb, download_obj, status_box],
        )

        refresh_history_btn.click(
            fn=on_refresh_history,
            inputs=[],
            outputs=[history_dropdown],
        )

    return demo, custom_css


if __name__ == "__main__":
    app, custom_css = create_ui()
    app.queue(max_size=2)
    app.launch(
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        inbrowser=True,
        css=custom_css,
    )
