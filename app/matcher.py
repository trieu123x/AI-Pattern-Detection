"""
matcher.py
==========
Template Matching đa tỷ lệ (multi-scale) và đa góc (multi-angle).

Thuật toán:
1. Với mỗi tổ hợp (scale, angle): resize → xoay → cv2.matchTemplate.
2. Các vị trí vượt ngưỡng được thu thập cùng điểm số.
3. NMS nội bộ (per-scale-angle) lọc box chồng nhau trước khi trả về.

Kết quả cuối được tổng hợp NMS lần thứ hai ở ``postprocess.py``.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default search space
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SCALES: list[float] = [round(0.9 + i * 0.1, 1) for i in range(8)]
# → [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

_DEFAULT_ANGLES: list[int] = list(range(0, 360, 45))
# → [0, 45, 90, 135, 180, 225, 270, 315]

# Số pixel nhỏ nhất mà pattern phải có sau khi scale để tiến hành matching
_MIN_TEMPLATE_SIZE: int = 5


# ──────────────────────────────────────────────────────────────────────────────
# Rotation helper
# ──────────────────────────────────────────────────────────────────────────────

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Xoay ảnh theo góc bất kỳ, canvas tự mở rộng để không bị cắt xén.

    - Góc 0°: trả về ảnh gốc không thay đổi (no-op).
    - Góc 90°/180°/270°: dùng ``cv2.rotate`` (nhanh, không mất chất lượng).
    - Các góc khác: dùng ``cv2.warpAffine`` với ``INTER_LINEAR``.
    - Background fill = 255 (trắng) để khớp với nền bản vẽ.

    Parameters
    ----------
    image : np.ndarray
        Ảnh đầu vào (uint8, grayscale hoặc BGR).
    angle : float
        Góc xoay theo chiều kim đồng hồ (độ).

    Returns
    -------
    np.ndarray
        Ảnh đã xoay, có thể lớn hơn ảnh gốc.

    Raises
    ------
    ValueError
        Nếu ``image`` là None hoặc mảng rỗng.
    """
    if image is None or image.size == 0:
        raise ValueError("rotate_image: ảnh đầu vào là None hoặc rỗng.")

    angle = float(angle) % 360

    if angle == 0:
        return image

    # Fast-path cho các góc vuông
    _ROTATE_CODES = {
        90.0:  cv2.ROTATE_90_CLOCKWISE,
        180.0: cv2.ROTATE_180,
        270.0: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if angle in _ROTATE_CODES:
        return cv2.rotate(image, _ROTATE_CODES[angle])

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)

    # Tính kích thước canvas mới để chứa toàn bộ ảnh sau khi xoay
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Dịch gốc toạ độ vào giữa canvas mới
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Core matching
# ──────────────────────────────────────────────────────────────────────────────

def match_pattern(
    pattern: np.ndarray,
    drawing: np.ndarray,
    threshold: float = 0.70,
    scales: Optional[list[float]] = None,
    angles: Optional[list[int]] = None,
    nms_iou_threshold: float = 0.3,
) -> list[dict]:
    """Multi-scale, multi-angle template matching.

    Parameters
    ----------
    pattern : np.ndarray
        Ảnh mẫu (template) đã tiền xử lý, grayscale uint8.
    drawing : np.ndarray
        Ảnh bản vẽ (search image) đã tiền xử lý, grayscale uint8.
    threshold : float, optional
        Ngưỡng tương đồng ``TM_CCOEFF_NORMED`` [0, 1]. Mặc định 0.70.
    scales : list[float] | None, optional
        Danh sách hệ số tỷ lệ cần thử. Mặc định dùng ``_DEFAULT_SCALES``.
    angles : list[int] | None, optional
        Danh sách góc xoay (độ) cần thử. Mặc định dùng ``_DEFAULT_ANGLES``.
    nms_iou_threshold : float, optional
        Ngưỡng IoU cho NMS nội bộ. Mặc định 0.3.

    Returns
    -------
    list[dict]
        Danh sách detection, mỗi phần tử gồm:
        ``{"bbox": [x, y, w, h], "score": float, "angle": float, "scale": float}``

    Raises
    ------
    ValueError
        Nếu ``pattern`` hoặc ``drawing`` là None / mảng rỗng.
        Nếu ``threshold`` nằm ngoài [0, 1].
    """
    if pattern is None or pattern.size == 0:
        raise ValueError("match_pattern: pattern là None hoặc rỗng.")
    if drawing is None or drawing.size == 0:
        raise ValueError("match_pattern: drawing là None hoặc rỗng.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"match_pattern: threshold phải trong [0,1], nhận {threshold}.")

    if scales is None:
        scales = _DEFAULT_SCALES
    if angles is None:
        angles = _DEFAULT_ANGLES

    boxes: list[dict] = []

    for scale in scales:
        if scale <= 0:
            logger.warning("match_pattern: bỏ qua scale không hợp lệ %s", scale)
            continue

        # Resize pattern theo hệ số scale
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized_pattern = cv2.resize(pattern, None, fx=scale, fy=scale, interpolation=interp)
        else:
            resized_pattern = pattern.copy()

        for angle in angles:
            rotated_pattern = rotate_image(resized_pattern, angle)
            ph, pw = rotated_pattern.shape[:2]

            # Bỏ qua nếu template quá nhỏ hoặc lớn hơn drawing
            if ph < _MIN_TEMPLATE_SIZE or pw < _MIN_TEMPLATE_SIZE:
                logger.debug("Bỏ qua: pattern quá nhỏ (scale=%.2f, angle=%d)", scale, angle)
                continue
            if ph >= drawing.shape[0] or pw >= drawing.shape[1]:
                logger.debug("Bỏ qua: pattern lớn hơn drawing (scale=%.2f, angle=%d)", scale, angle)
                continue

            # Dense template matching
            res = cv2.matchTemplate(drawing, rotated_pattern, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= threshold)

            if len(xs) == 0:
                continue

            rects = [[int(x), int(y), int(pw), int(ph)] for x, y in zip(xs, ys)]
            scores = [float(res[y, x]) for x, y in zip(xs, ys)]

            # NMS nội bộ per-(scale, angle) để giảm số candidate trước khi gộp
            indices = cv2.dnn.NMSBoxes(rects, scores, threshold, nms_iou_threshold)
            if len(indices) == 0:
                continue

            for i in indices.flatten():
                x, y, w, h = rects[i]
                boxes.append({
                    "bbox":  [x, y, w, h],
                    "score": round(scores[i], 4),
                    "angle": float(angle),
                    "scale": float(scale),
                })

    logger.info("match_pattern: tìm được %d candidate (trước NMS toàn cục)", len(boxes))
    return boxes
