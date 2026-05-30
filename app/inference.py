"""
inference.py
============
Điểm vào (entry point) duy nhất của pipeline phát hiện mẫu.

Flow:
    preprocess_pattern → crop_pattern
    preprocess_image
    match_pattern  (Template Matching multi-scale × multi-angle)
    apply_nms      (NMS toàn cục)
    draw_boxes     (Visualisation)
"""

import logging
import os

import cv2
import numpy as np

from app.preprocess  import preprocess_pattern, preprocess_image, crop_pattern
from app.matcher     import match_pattern
from app.postprocess import apply_nms
from app.visualize   import draw_boxes

logger = logging.getLogger(__name__)


def _load_image(path: str, label: str = "ảnh") -> np.ndarray:
    """Đọc ảnh từ đường dẫn và kiểm tra tính hợp lệ.

    Parameters
    ----------
    path : str
        Đường dẫn file ảnh.
    label : str, optional
        Tên gọi ảnh, dùng trong thông báo lỗi. Mặc định ``"ảnh"``.

    Returns
    -------
    np.ndarray
        Ảnh đọc bởi ``cv2.imread`` (BGR uint8).

    Raises
    ------
    FileNotFoundError
        Nếu file không tồn tại.
    ValueError
        Nếu ``cv2.imread`` trả về None (không đọc được ảnh).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy {label}: '{path}'")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(
            f"Không thể đọc {label} '{path}'. "
            "Kiểm tra định dạng file (JPEG, PNG, BMP…)."
        )
    return img


def run_inference(
    pattern_path: str,
    drawing_path: str,
    tm_threshold: float = 0.60,
    scales: list[float] | None = None,
    angles: list[int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Chạy toàn bộ pipeline phát hiện mẫu.

    Parameters
    ----------
    pattern_path : str
        Đường dẫn tới ảnh mẫu (template).
    drawing_path : str
        Đường dẫn tới ảnh bản vẽ (search image).
    tm_threshold : float, optional
        Ngưỡng Template Matching ``TM_CCOEFF_NORMED`` [0, 1]. Mặc định 0.60.
    scales : list[float] | None, optional
        Danh sách hệ số scale cần thử. Nếu ``None``, dùng mặc định.
    angles : list[int] | None, optional
        Danh sách góc xoay (độ) cần thử. Nếu ``None``, dùng mặc định.

    Returns
    -------
    vis : np.ndarray
        Ảnh BGR đã vẽ bounding box (dùng để hiển thị).
    boxes : list[dict]
        Danh sách detection cuối cùng sau NMS.

    Raises
    ------
    FileNotFoundError
        Nếu một trong hai file ảnh không tồn tại.
    ValueError
        Nếu không thể đọc ảnh hoặc các tham số không hợp lệ.
    """
    logger.info("run_inference: pattern='%s', drawing='%s'", pattern_path, drawing_path)

    # ── 1. Đọc ảnh ───────────────────────────────────────────────────────────
    raw_pattern = _load_image(pattern_path, "pattern")
    raw_drawing = _load_image(drawing_path, "drawing")

    # ── 2. Tiền xử lý ────────────────────────────────────────────────────────
    pattern = preprocess_pattern(raw_pattern)
    pattern = crop_pattern(pattern)
    drawing = preprocess_image(raw_drawing)

    # ── 3. Template Matching (multi-scale × multi-angle) ──────────────────────
    boxes = match_pattern(
        pattern, drawing,
        threshold=tm_threshold,
        scales=scales,
        angles=angles,
    )

    # ── 4. NMS toàn cục ───────────────────────────────────────────────────────
    boxes = apply_nms(boxes)

    # ── 5. Visualisation ──────────────────────────────────────────────────────
    vis = draw_boxes(drawing, boxes)

    logger.info("run_inference: kết thúc, %d detection(s).", len(boxes))
    return vis, boxes
