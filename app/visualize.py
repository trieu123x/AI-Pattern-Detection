"""
visualize.py
============
Hiển thị kết quả phát hiện mẫu lên ảnh.

Hiệu ứng "spotlight": vùng ngoài bounding box bị làm tối (35% độ sáng),
trong khi các vùng phát hiện được giữ nguyên độ sáng gốc, tạo điểm nhấn
trực quan rõ ràng.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BOX_COLOR   = (0, 0, 255)   # BGR: đỏ
_LABEL_COLOR = (0, 0, 200)   # BGR: đỏ đậm (nền chữ)
_TEXT_COLOR  = (255, 255, 255)
_DIM_FACTOR  = 0.35          # 35% độ sáng cho vùng nền
_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.45
_BOX_THICK   = 2


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def draw_boxes(
    img: np.ndarray,
    boxes: list[dict],
) -> np.ndarray:
    """Vẽ bounding box và score lên ảnh với hiệu ứng spotlight.

    Parameters
    ----------
    img : np.ndarray
        Ảnh gốc (uint8), có thể grayscale hoặc BGR.
    boxes : list[dict]
        Danh sách detection. Mỗi phần tử cần có:
        - ``"bbox"``: ``[x, y, w, h]`` (int).
        - ``"score"``: float trong [0, 1].

    Returns
    -------
    np.ndarray
        Ảnh BGR đã vẽ bounding box và score.

    Raises
    ------
    ValueError
        Nếu ``img`` là None hoặc mảng rỗng.
    """
    if img is None or img.size == 0:
        raise ValueError("draw_boxes: ảnh đầu vào là None hoặc rỗng.")

    # Đảm bảo đầu ra luôn là ảnh BGR để vẽ màu
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    if not boxes:
        logger.debug("draw_boxes: không có detection, trả về ảnh gốc.")
        return out

    # ── Hiệu ứng spotlight ────────────────────────────────────────────────────
    dark   = (out * _DIM_FACTOR).astype(np.uint8)
    result = dark.copy()

    for item in boxes:
        x, y, w, h = item["bbox"]
        score       = float(item["score"])

        # Giới hạn tọa độ trong ảnh để tránh out-of-bound
        x1, y1 = max(0, x), max(0, y)
        x2 = min(out.shape[1], x + w)
        y2 = min(out.shape[0], y + h)

        if x2 <= x1 or y2 <= y1:
            logger.warning("draw_boxes: bỏ qua box ngoài biên ảnh %s", item["bbox"])
            continue

        # Khôi phục độ sáng gốc bên trong bbox
        result[y1:y2, x1:x2] = out[y1:y2, x1:x2]

        # Vẽ viền đỏ
        cv2.rectangle(result, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICK)

        # Nhãn score với nền pill
        label = f"{score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        lx = x1
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(
            result,
            (lx, ly - th - baseline - 2),
            (lx + tw + 4, ly + baseline),
            _LABEL_COLOR, -1,
        )
        cv2.putText(
            result, label,
            (lx + 2, ly - 2),
            _FONT, _FONT_SCALE, _TEXT_COLOR, 1, cv2.LINE_AA,
        )

    logger.debug("draw_boxes: đã vẽ %d bounding box.", len(boxes))
    return result
