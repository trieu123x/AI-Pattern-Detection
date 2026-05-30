"""
preprocess.py
=============
Các hàm tiền xử lý ảnh cho pipeline phát hiện mẫu.

Quy ước:
- Tất cả hàm nhận và trả về numpy.ndarray kiểu uint8.
- Ảnh đầu ra luôn là grayscale (2-D array, shape H×W).
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang grayscale nếu chưa phải.

    Hỗ trợ cả BGR (3-kênh) và BGRA (4-kênh).

    Parameters
    ----------
    img : np.ndarray
        Ảnh đầu vào (uint8).

    Returns
    -------
    np.ndarray
        Ảnh grayscale 2-D.

    Raises
    ------
    ValueError
        Nếu ảnh có số kênh không được hỗ trợ.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(
        f"Ảnh có shape không được hỗ trợ: {img.shape}. "
        "Chỉ hỗ trợ grayscale (H,W), BGR (H,W,3) và BGRA (H,W,4)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh bản vẽ lớn (search image).

    Pipeline:
    1. Chuyển sang grayscale.
    2. Morphological Opening (kernel 2×2) để loại bỏ nhiễu nhỏ.

    Parameters
    ----------
    img : np.ndarray
        Ảnh đầu vào (uint8), có thể là BGR hoặc grayscale.

    Returns
    -------
    np.ndarray
        Ảnh grayscale đã tiền xử lý.

    Raises
    ------
    ValueError
        Nếu ``img`` là None hoặc mảng rỗng.
    """
    if img is None or img.size == 0:
        raise ValueError("preprocess_image: ảnh đầu vào là None hoặc rỗng.")

    gray = _ensure_grayscale(img)

    # Morphological Opening loại bỏ noise nhỏ, giữ nguyên đường nét lớn
    kernel = np.ones((2, 2), np.uint8)
    result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    logger.debug("preprocess_image: %s → grayscale %s", img.shape, result.shape)
    return result


def preprocess_pattern(img: np.ndarray) -> np.ndarray:
    """Tiền xử lý ảnh mẫu (template) nhỏ.

    Không áp dụng Morphological Opening vì pattern nhỏ — thao tác đó sẽ
    xóa các nét mảnh (ví dụ: cạnh hộp điện trở).

    Pipeline:
    1. Chuyển sang grayscale.
    2. Binary threshold (ngưỡng 200) để có nét đen trên nền trắng rõ ràng.

    Parameters
    ----------
    img : np.ndarray
        Ảnh mẫu đầu vào (uint8).

    Returns
    -------
    np.ndarray
        Ảnh grayscale nhị phân.

    Raises
    ------
    ValueError
        Nếu ``img`` là None hoặc mảng rỗng.
    """
    if img is None or img.size == 0:
        raise ValueError("preprocess_pattern: ảnh đầu vào là None hoặc rỗng.")

    gray = _ensure_grayscale(img)
    _, result = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    logger.debug("preprocess_pattern: %s → binary %s", img.shape, result.shape)
    return result


def crop_pattern(img: np.ndarray, padding: int = 4) -> np.ndarray:
    """Cắt vùng nội dung của ảnh mẫu, loại bỏ viền trắng thừa.

    Algorithm:
    1. Chuyển sang grayscale.
    2. Invert + threshold → tìm pixel nội dung (màu đen trên nền trắng).
    3. Tính bounding rect của toàn bộ pixel nội dung.
    4. Crop + thêm ``padding`` pixel mỗi phía.

    Nếu không tìm thấy pixel nội dung, trả về ảnh gốc nguyên vẹn.

    Parameters
    ----------
    img : np.ndarray
        Ảnh mẫu grayscale hoặc BGR (uint8).
    padding : int, optional
        Số pixel padding mỗi phía. Mặc định 4.

    Returns
    -------
    np.ndarray
        Ảnh đã crop (grayscale).

    Raises
    ------
    ValueError
        Nếu ``img`` là None hoặc mảng rỗng.
    """
    if img is None or img.size == 0:
        raise ValueError("crop_pattern: ảnh đầu vào là None hoặc rỗng.")
    if padding < 0:
        raise ValueError(f"crop_pattern: padding phải >= 0, nhận được {padding}.")

    gray = _ensure_grayscale(img)

    # Invert: pixel đen (nội dung) → trắng → dễ tìm bằng findNonZero
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    if coords is None:
        logger.warning("crop_pattern: không tìm thấy nội dung, trả về ảnh gốc.")
        return gray

    x, y, w, h = cv2.boundingRect(coords)

    # Áp dụng padding có kiểm tra biên
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)

    cropped = gray[y1:y2, x1:x2]
    logger.debug(
        "crop_pattern: crop (%d,%d,%d,%d) + pad=%d → shape %s",
        x, y, w, h, padding, cropped.shape,
    )
    return cropped