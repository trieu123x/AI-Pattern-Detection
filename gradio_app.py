import gradio as gr

from app.inference import run_inference
from app.matcher import _DEFAULT_SCALES, _DEFAULT_ANGLES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_path(img_input) -> str:
    """Gradio đôi khi trả về dict FileData thay vì string filepath."""
    if isinstance(img_input, dict):
        return img_input.get("path") or img_input.get("name", "")
    return img_input or ""


def _build_scales(s_min: float, s_max: float, s_step: float) -> list[float]:
    """Tạo danh sách scale từ min/max/step, làm tròn 3 chữ số thập phân."""
    if s_step <= 0 or s_min > s_max:
        return []
    result, v = [], s_min
    while v <= s_max + 1e-9:
        result.append(round(v, 3))
        v += s_step
    return result


def _build_angles(a_min: int, a_max: int, a_step: int) -> list[int]:
    """Tạo danh sách angle từ min/max/step (đơn vị độ)."""
    if a_step <= 0 or a_min > a_max:
        return []
    return list(range(int(a_min), int(a_max) + 1, int(a_step)))


def _preview_scales(s_min, s_max, s_step) -> str:
    vals = _build_scales(s_min, s_max, s_step)
    if not vals:
        return "⚠️ Danh sách trống – kiểm tra lại min / max / step"
    return f"**Scales ({len(vals)} mức):** `{vals}`"


def _preview_angles(a_min, a_max, a_step) -> str:
    vals = _build_angles(a_min, a_max, a_step)
    if not vals:
        return "⚠️ Danh sách trống – kiểm tra lại min / max / step"
    return f"**Angles ({len(vals)} góc):** `{vals}`"


def infer(
    pattern, drawing, tm_threshold,
    use_custom,
    s_min, s_max, s_step,
    a_min, a_max, a_step,
):
    pattern_path = _get_path(pattern)
    drawing_path = _get_path(drawing)

    scales = _build_scales(s_min, s_max, s_step) if use_custom else None
    angles = _build_angles(a_min, a_max, a_step) if use_custom else None

    vis, boxes = run_inference(
        pattern_path, drawing_path,
        tm_threshold=tm_threshold,
        scales=scales or None,
        angles=angles or None,
    )
    summary = {"total_detections": len(boxes), "boxes": boxes}
    return vis, summary


# ── UI ───────────────────────────────────────────────────────────────────────

# Defaults cho advanced panel
_S_MIN_DEF  = min(_DEFAULT_SCALES)
_S_MAX_DEF  = max(_DEFAULT_SCALES)
_S_STEP_DEF = round(_DEFAULT_SCALES[1] - _DEFAULT_SCALES[0], 3) if len(_DEFAULT_SCALES) > 1 else 0.1

_A_MIN_DEF  = min(_DEFAULT_ANGLES)
_A_MAX_DEF  = max(_DEFAULT_ANGLES)
_A_STEP_DEF = (_DEFAULT_ANGLES[1] - _DEFAULT_ANGLES[0]) if len(_DEFAULT_ANGLES) > 1 else 45


with gr.Blocks(title="Pattern Detection") as demo:
    gr.Markdown(
        "## Pattern Detection\n"
        "**Bước 1:** Template Matching (multi-scale, multi-angle) đề xuất candidate boxes — "
        "**Bước 2:** NMS gộp các box chồng nhau"
    )

    # ── Quick-test buttons ────────────────────────────────────────────────────
    gr.Markdown("### Quick Test")
    with gr.Row():
        test1_btn = gr.Button("Test 1", variant="secondary", scale=1)
        test2_btn = gr.Button("Test 2", variant="secondary", scale=1)
        gr.Column(scale=8)  # spacer

    # Pattern preset rows (hidden by default, shown after clicking a test btn)
    with gr.Row(visible=False) as pattern_row_1:
        gr.Markdown("**Pattern cho Test 1:**")
        p1_1_btn = gr.Button("Pattern 1 (1.1)", scale=1)
        p1_2_btn = gr.Button("Pattern 2 (1.2)", scale=1)
        p1_3_btn = gr.Button("Pattern 3 (1.3)", scale=1)

    with gr.Row(visible=False) as pattern_row_2:
        gr.Markdown("**Pattern cho Test 2:**")
        p2_1_btn = gr.Button("Pattern 1 (4.1)", scale=1)
        p2_2_btn = gr.Button("Pattern 2 (4.2)", scale=1)

    gr.Markdown("---")

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────
        with gr.Column(scale=1):
            pattern_input = gr.Image(type="filepath", label="Pattern (template)")
            drawing_input = gr.Image(type="filepath", label="Drawing (search image)")

            tm_slider = gr.Slider(
                minimum=0.30, maximum=0.95, value=0.70, step=0.05,
                label="Template Match Threshold",
                info="Thấp hơn = nhiều candidate hơn; Cao hơn = chính xác hơn",
            )

            # ── Advanced Settings ─────────────────────────────────────────
            with gr.Accordion("Advanced Settings", open=False):
                use_custom_cb = gr.Checkbox(
                    label="Bật tuỳ chỉnh (bỏ chọn = dùng mặc định)",
                    value=False,
                )

                gr.Markdown("### Scale")
                with gr.Row():
                    s_min_num  = gr.Number(value=_S_MIN_DEF,  label="Min",  precision=3)
                    s_max_num  = gr.Number(value=_S_MAX_DEF,  label="Max",  precision=3)
                    s_step_num = gr.Number(value=_S_STEP_DEF, label="Step", precision=3)
                scale_preview = gr.Markdown(
                    _preview_scales(_S_MIN_DEF, _S_MAX_DEF, _S_STEP_DEF)
                )

                gr.Markdown("### Angle (°)")
                with gr.Row():
                    a_min_num  = gr.Number(value=_A_MIN_DEF,  label="Min",  precision=0)
                    a_max_num  = gr.Number(value=_A_MAX_DEF,  label="Max",  precision=0)
                    a_step_num = gr.Number(value=_A_STEP_DEF, label="Step", precision=0)
                angle_preview = gr.Markdown(
                    _preview_angles(_A_MIN_DEF, _A_MAX_DEF, _A_STEP_DEF)
                )

            with gr.Row():
                clear_btn  = gr.Button("Clear",  variant="secondary")
                submit_btn = gr.Button("Submit", variant="primary")

        # ── Right column: outputs ─────────────────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(label="Detection Result")
            output_json  = gr.JSON(label="Detections")

    # ── Live preview wiring ───────────────────────────────────────────────
    for num_inp in [s_min_num, s_max_num, s_step_num]:
        num_inp.change(
            fn=_preview_scales,
            inputs=[s_min_num, s_max_num, s_step_num],
            outputs=scale_preview,
        )

    for num_inp in [a_min_num, a_max_num, a_step_num]:
        num_inp.change(
            fn=_preview_angles,
            inputs=[a_min_num, a_max_num, a_step_num],
            outputs=angle_preview,
        )

    # ── Submit wiring ─────────────────────────────────────────────────────
    _infer_inputs = [
        pattern_input, drawing_input, tm_slider,
        use_custom_cb,
        s_min_num, s_max_num, s_step_num,
        a_min_num, a_max_num, a_step_num,
    ]

    submit_btn.click(
        fn=lambda: (None, None),
        outputs=[output_image, output_json],
        queue=False,
    ).then(
        fn=infer,
        inputs=_infer_inputs,
        outputs=[output_image, output_json],
    )

    clear_btn.click(
        fn=lambda: (None, None, None, None),
        outputs=[pattern_input, drawing_input, output_image, output_json],
        queue=False,
    )

    # ── Quick-test wiring ─────────────────────────────────────────────────────
    _DRAW_1 = "img/draw/1.png"
    _DRAW_4 = "img/draw/4.png"

    _PAT_1_1 = "img/pattern/1.1.png"
    _PAT_1_2 = "img/pattern/1.2.png"
    _PAT_1_3 = "img/pattern/1.3.png"
    _PAT_4_1 = "img/pattern/4.1.png"
    _PAT_4_2 = "img/pattern/4.2.png"

    # Test 1: load draw/1.png + show pattern row 1, hide row 2
    test1_btn.click(
        fn=lambda: (_DRAW_1, gr.update(visible=True), gr.update(visible=False)),
        outputs=[drawing_input, pattern_row_1, pattern_row_2],
        queue=False,
    )

    # Test 2: load draw/4.png + show pattern row 2, hide row 1
    test2_btn.click(
        fn=lambda: (_DRAW_4, gr.update(visible=False), gr.update(visible=True)),
        outputs=[drawing_input, pattern_row_1, pattern_row_2],
        queue=False,
    )

    # Pattern buttons for Test 1
    p1_1_btn.click(fn=lambda: _PAT_1_1, outputs=pattern_input, queue=False)
    p1_2_btn.click(fn=lambda: _PAT_1_2, outputs=pattern_input, queue=False)
    p1_3_btn.click(fn=lambda: _PAT_1_3, outputs=pattern_input, queue=False)

    # Pattern buttons for Test 2
    p2_1_btn.click(fn=lambda: _PAT_4_1, outputs=pattern_input, queue=False)
    p2_2_btn.click(fn=lambda: _PAT_4_2, outputs=pattern_input, queue=False)


if __name__ == "__main__":
    demo.launch()
