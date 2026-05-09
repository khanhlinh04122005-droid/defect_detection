"""
app.py — Gradio UI cho Fabric Defect Detection Pipeline

Cách chạy:
    # GPU machine
    set PYTHONPATH=D:\Projects\defect_detection
    python app.py

    # CPU machine (Dell Vostro) — cần Ollama đang chạy
    python app.py --ollama

    # Colab — tạo public URL
    python app.py --share
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ollama",  action="store_true", help="Dùng Ollama thay InternVL2")
parser.add_argument("--share",   action="store_true", help="Tạo public Gradio link")
parser.add_argument("--port",    type=int, default=7860)
parser.add_argument("--category", default="tsfabric_T1")
args, _ = parser.parse_known_args()

USE_OLLAMA = args.ollama or not torch.cuda.is_available()
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORIES = ["tsfabric_T1", "tsfabric_T2", "tsfabric_T3"]

print(f"[App] Device: {DEVICE} | Chat backend: {'Ollama' if USE_OLLAMA else 'InternVL2'}")

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_engine     = None
_vlm_model  = None   # InternVL2Wrapper (GPU only)
_ollama_inf = None   # VLMInferenceOllama (CPU)


def _get_engine(category: str):
    from stage4_decision.engine import PipelineEngine
    global _engine
    if _engine is None or _engine.category != category:
        _engine = PipelineEngine(category=category, device=DEVICE)
    return _engine


def _get_chat_backend():
    global _vlm_model, _ollama_inf
    if USE_OLLAMA:
        if _ollama_inf is None:
            from stage3_vlm.inference_ollama import VLMInferenceOllama
            _ollama_inf = VLMInferenceOllama()
        return _ollama_inf
    else:
        if _vlm_model is None:
            from stage3_vlm.model import InternVL2Wrapper
            from configs.stage3_config import LORA_WEIGHTS_PATH
            _vlm_model = InternVL2Wrapper(
                device=DEVICE, use_lora=True,
                lora_weights=LORA_WEIGHTS_PATH, force_fp16=True,
            )
            _vlm_model.eval_mode()
        return _vlm_model


# ── Pipeline analysis ─────────────────────────────────────────────────────────

def run_analysis(image_pil, category):
    if image_pil is None:
        return None, "⚠️ Chưa upload ảnh.", "", {}

    # Lưu ảnh tạm để pipeline đọc
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        image_pil.save(tmp_path, "JPEG")

    try:
        engine = _get_engine(category)
        result = engine.run(tmp_path)
    except Exception as e:
        return None, f"❌ Lỗi pipeline: {e}", "", {}
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Tạo overlay image
    img_np   = np.array(image_pil.convert("RGB"))
    overlay  = _make_overlay(img_np, result)

    # Verdict text
    decision = result.get("decision", {})
    verdict  = decision.get("verdict", "Unknown")
    priority = decision.get("priority_label", "")
    conf     = decision.get("confidence", 0)
    emoji    = "🔴 FAIL" if verdict == "Fail" else "🟢 PASS"
    verdict_md = (
        f"## {emoji}\n"
        f"**Priority:** {priority} | **Confidence:** {conf:.0%}\n\n"
        + ("\n".join(f"- {r}" for r in decision.get("reasons", [])))
    )

    # Stage details
    s1 = result.get("stage1", {})
    s2 = result.get("stage2", {})
    s3 = result.get("stage3", {})
    details_md = (
        f"### Stage 1 — Anomaly Detection\n"
        f"Score: `{s1.get('score', 0):.3f}` | Threshold: `{s1.get('threshold', 0):.3f}` | "
        f"Vote: **{s1.get('prediction', '-')}** | ⏱ {s1.get('elapsed_ms', 0)}ms\n\n"
        f"### Stage 2 — Segmentation (SAM2)\n"
        f"Area: `{s2.get('area_ratio', 0)*100:.1f}%` | "
        f"SAM conf: `{s2.get('confidence', 0):.2f}` | ⏱ {s2.get('elapsed_ms', 0)}ms\n\n"
        f"### Stage 3 — VLM\n"
        f"Type: `{s3.get('defect_type', '-')}` | "
        f"Severity: `{s3.get('severity', '-')}` | "
        f"Conf: `{s3.get('confidence', 0):.2f}` | ⏱ {s3.get('elapsed_ms', 0)}ms\n\n"
        f"**Caption:** {s3.get('caption', 'N/A')}"
    )

    return overlay, verdict_md, details_md, result


def _make_overlay(img_np: np.ndarray, result: dict) -> np.ndarray:
    import cv2
    overlay = img_np.copy().astype(np.float32)
    mask_np = result.get("stage2", {}).get("mask")
    if mask_np is not None and isinstance(mask_np, np.ndarray) and mask_np.any():
        red = np.zeros_like(img_np, dtype=np.float32)
        red[mask_np] = [255, 60, 60]
        overlay = overlay * 0.6 + red * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        mask_u8 = mask_np.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)
        return overlay
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ── Chat ──────────────────────────────────────────────────────────────────────

def chat_fn(message, history, pipeline_result, image_pil):
    if not pipeline_result:
        yield "⚠️ Hãy upload và phân tích ảnh trước khi chat."
        return

    if image_pil is None:
        yield "⚠️ Không tìm thấy ảnh."
        return

    img_np = np.array(image_pil.convert("RGB"))

    # Build context từ pipeline result
    decision = pipeline_result.get("decision", {})
    s1       = pipeline_result.get("stage1", {})
    s3       = pipeline_result.get("stage3", {})

    context = (
        f"You are a fabric quality inspector assistant. "
        f"A fabric image was just analyzed with these results:\n"
        f"- Verdict: {decision.get('verdict', 'Unknown')}\n"
        f"- Priority: {decision.get('priority_label', '-')}\n"
        f"- Defect type: {s3.get('defect_type', 'unknown')}\n"
        f"- Severity: {s3.get('severity', 'unknown')}\n"
        f"- Location: {s3.get('location', 'unknown')}\n"
        f"- Caption: {s3.get('caption', 'N/A')}\n"
        f"- Anomaly score: {s1.get('score', 0):.1f} (threshold: {s1.get('threshold', 0):.1f})\n\n"
        f"Answer the user's question concisely. "
        f"Reply in the same language as the question (Vietnamese or English).\n\n"
        f"Question: {message}"
    )

    backend = _get_chat_backend()

    if USE_OLLAMA:
        from stage3_vlm.inference_ollama import _encode_image, _ollama_chat, DEFAULT_MODEL
        img_b64  = _encode_image(img_np)
        response = _ollama_chat(DEFAULT_MODEL, context, img_b64, timeout=120)
        yield response
    else:
        # InternVL2 GPU mode
        pixel_values = backend._build_pixel_values(img_np)
        gen_cfg = dict(max_new_tokens=256, do_sample=False)
        response = backend.model.chat(
            backend.tokenizer, pixel_values, context, gen_cfg
        )
        yield response


# ── Gradio UI ─────────────────────────────────────────────────────────────────

TITLE = "🔍 Fabric Defect Detection"
CSS   = """
.verdict-pass { color: #28a745; font-size: 1.3em; }
.verdict-fail { color: #dc3545; font-size: 1.3em; }
.stage-box { background: #f8f9fa; border-radius: 8px; padding: 12px; }
"""

with gr.Blocks(title=TITLE, css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        f"Backend: **{'Ollama (CPU)' if USE_OLLAMA else 'InternVL2 (GPU)'}** | "
        f"Device: **{DEVICE}**"
    )

    # ── State ──
    pipeline_result = gr.State({})

    # ── Row 1: Input + Results ──
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload ảnh vải")
            category_dd = gr.Dropdown(
                choices=CATEGORIES, value=args.category, label="Category"
            )
            analyze_btn = gr.Button("🔬 Phân tích", variant="primary", size="lg")

        with gr.Column(scale=1):
            overlay_out = gr.Image(label="Mask Overlay", interactive=False)
            verdict_out = gr.Markdown("*Chưa phân tích*")

    # ── Row 2: Stage details ──
    with gr.Accordion("Chi tiết từng Stage", open=False):
        details_out = gr.Markdown()

    gr.Divider()

    # ── Chat ──
    gr.Markdown("### 💬 Chat về kết quả phân tích")
    gr.Markdown(
        "*Sau khi phân tích ảnh, bạn có thể hỏi thêm về lỗi, "
        "mức độ ảnh hưởng, cách xử lý...*"
    )

    chatbot = gr.Chatbot(height=320, label="Chat với AI")
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="Ví dụ: Lỗi này có nghiêm trọng không? / What caused this defect?",
            label="", scale=5, lines=1,
        )
        chat_btn = gr.Button("Gửi", variant="secondary", scale=1)

    clear_btn = gr.Button("🗑 Xóa chat", size="sm")

    # ── Event handlers ──
    def on_analyze(image, category):
        overlay, verdict, details, result = run_analysis(image, category)
        return overlay, verdict, details, result

    analyze_btn.click(
        fn=on_analyze,
        inputs=[image_input, category_dd],
        outputs=[overlay_out, verdict_out, details_out, pipeline_result],
    )

    def on_chat(message, history, result, image):
        if not message.strip():
            return history, ""
        history = history or []
        history.append([message, None])
        response = ""
        for chunk in chat_fn(message, history, result, image):
            response = chunk
        history[-1][1] = response
        return history, ""

    chat_btn.click(
        fn=on_chat,
        inputs=[chat_input, chatbot, pipeline_result, image_input],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        fn=on_chat,
        inputs=[chat_input, chatbot, pipeline_result, image_input],
        outputs=[chatbot, chat_input],
    )
    clear_btn.click(lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=not args.share,
    )
