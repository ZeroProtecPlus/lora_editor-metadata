from modules import script_callbacks
import gradio as gr
import json
import sys
import io
import subprocess
import tempfile
from pathlib import Path
from safetensors_worker import PrintMetadata

class Context:
    def __init__(self):
        self.obj = {'quiet': True, 'parse_more': True}

ctx = Context()

def debug_log(message: str):
    print(f"[DEBUG] {message}")

def load_metadata(file_path: str) -> tuple:
    try:
        debug_log(f"Loading file: {file_path}")
        
        if not file_path:
            return {"status": "Awaiting input"}, {}, "", "", ""

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        exit_code = PrintMetadata(ctx.obj, file_path.name)
        sys.stdout = old_stdout
        
        metadata_str = buffer.getvalue().strip()
        
        if exit_code != 0:
            error_msg = f"Error code {exit_code}"
            return {"error": error_msg}, {}, "", error_msg, ""

        try:
            full_metadata = json.loads(metadata_str)
        except json.JSONDecodeError:
            error_msg = "Invalid metadata structure"
            return {"error": error_msg}, {}, "", error_msg, ""

        training_params = full_metadata.get("__metadata__", {})
        key_metrics = {
            key: training_params.get(key, "N/A")
            for key in [
                "ss_optimizer", "ss_num_epochs", "ss_unet_lr",
                "ss_text_encoder_lr", "ss_steps"
            ]
        }
        
        return full_metadata, key_metrics, json.dumps(full_metadata, indent=2), "", file_path.name
    
    except Exception as e:
        return {"error": str(e)}, {}, "", str(e), ""

def validate_json(edited_json: str) -> tuple:
    try:
        return True, json.loads(edited_json), ""
    except Exception as e:
        return False, None, str(e)

def update_metadata(edited_json: str) -> tuple:
    try:
        modified_data = json.loads(edited_json)
        metadata = modified_data.get("__metadata__", {})
        
        key_fields = {
            param: metadata.get(param, "N/A")
            for param in [
                "ss_optimizer", "ss_num_epochs", "ss_unet_lr",
                "ss_text_encoder_lr", "ss_steps"
            ]
        }
        return key_fields, modified_data, ""
    except:
        return gr.update(), gr.update(), ""

def save_metadata(edited_json: str, source_file: str, output_name: str) -> tuple:
    debug_log("Initiating save process")
    try:
        if not source_file:
            return None, "No source file provided"

        is_valid, parsed_data, error = validate_json(edited_json)
        if not is_valid:
            return None, f"Validation error: {error}"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(parsed_data, tmp, indent=2)
            temp_path = tmp.name

        source_path = Path(source_file)
        
        if output_name.strip():
            base_name = output_name.strip()
            if not base_name.endswith(".safetensors"):
                base_name += ".safetensors"
        else:
            base_name = f"{source_path.stem}_modified.safetensors"
        
        output_path = Path(base_name)
        version = 1
        while output_path.exists():
            output_path = Path(f"{source_path.stem}_modified_{version}.safetensors")
            version += 1

        cmd = [
            sys.executable,
            "safetensors_util.py",
            "writemd",
            source_file,
            temp_path,
            str(output_path),
            "-f"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        Path(temp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            error_msg = f"Save failure: {result.stderr}"
            return None, error_msg

        return str(output_path), ""
    
    except Exception as e:
        return None, f"Critical error: {str(e)}"

def on_ui_tabs():
    with gr.Blocks(title="LoRA Metadata Editor", analytics_enabled=False) as lora_editor:
        gr.Markdown("# LoRA Metadata Editor")
        
        with gr.Tabs():
            with gr.Tab("Metdata Viewer"):
                gr.Markdown("### LoRa Upload")
                file_input = gr.File(
                    file_types=[".safetensors"],
                    show_label=False
                )
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Full Metadata")
                        full_viewer = gr.JSON(show_label=False)
                    
                    with gr.Column():
                        gr.Markdown("### Key Metrics")
                        key_viewer = gr.JSON(show_label=False)

            with gr.Tab("Edit Metadata"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### JSON Workspace")
                        metadata_editor = gr.Textbox(
                            lines=25,
                            show_label=False,
                            placeholder="Edit metadata JSON here"
                        )
                        gr.Markdown("### Output Name")
                        filename_input = gr.Textbox(
                            placeholder="Leave empty for auto-naming",
                            show_label=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Live Preview")
                        modified_viewer = gr.JSON(show_label=False)
                        save_btn = gr.Button("💾 Save Metadata", variant="primary")
                        gr.Markdown("### Download Modified LoRa") 
                        output_file = gr.File(
                            visible=False,
                            show_label=False
                        )

        status_display = gr.HTML(visible=False)
        source_tracker = gr.State()

        file_input.upload(
            load_metadata,
            inputs=file_input,
            outputs=[full_viewer, key_viewer, metadata_editor, status_display, source_tracker]
        )

        metadata_editor.change(
            update_metadata,
            inputs=metadata_editor,
            outputs=[key_viewer, modified_viewer, status_display]
        )

        save_btn.click(
            save_metadata,
            inputs=[metadata_editor, source_tracker, filename_input],
            outputs=[output_file, status_display],
        ).then(
            lambda x: gr.File(value=x, visible=True),
            inputs=output_file,
            outputs=output_file
        )

    return [(lora_editor, "Lora Metadata Editor", "lora_editor")]