"""
postprocess.py
==============
Hậu xử lý danh sách detection từ matcher.

Hiện tại gồm một bước duy nhất: Non-Maximum Suppression (NMS) toàn cục
để loại bỏ các bounding box chồng nhau từ nhiều scale/angle khác nhau.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_nms(
    boxes: list[dict],
    score_threshold: float = 0.50,
    nms_threshold: float = 0.30,
) -> list[dict]:
    """Áp dụng Non-Maximum Suppression lên toàn bộ danh sách detection.

    Parameters
    ----------
    boxes : list[dict]
        Danh sách detection, mỗi phần tử phải có khoá ``"bbox"``
        (``[x, y, w, h]`` dạng int) và ``"score"`` (float trong [0, 1]).
    score_threshold : float, optional
        Chỉ giữ lại box có score >= ngưỡng này. Mặc định 0.50.
    nms_threshold : float, optional
        Ngưỡng IoU; các box chồng nhau vượt ngưỡng này sẽ bị loại bỏ.
        Mặc định 0.30.

    Returns
    -------
    list[dict]
        Danh sách detection sau NMS, thứ tự giữ nguyên theo index gốc.

    Raises
    ------
    ValueError
        Nếu ``score_threshold`` hoặc ``nms_threshold`` nằm ngoài [0, 1].
    """
    if not (0.0 <= score_threshold <= 1.0):
        raise ValueError(
            f"apply_nms: score_threshold phải trong [0,1], nhận {score_threshold}."
        )
    if not (0.0 <= nms_threshold <= 1.0):
        raise ValueError(
            f"apply_nms: nms_threshold phải trong [0,1], nhận {nms_threshold}."
        )

    if not boxes:
        logger.debug("apply_nms: danh sách đầu vào rỗng, trả về [].")
        return []

    bboxes = [b["bbox"] for b in boxes]
    scores = [float(b["score"]) for b in boxes]

    indices = cv2.dnn.NMSBoxes(
        bboxes,
        scores,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    if len(indices) == 0:
        logger.info("apply_nms: tất cả box bị loại bỏ bởi NMS.")
        return []

    result = [boxes[i] for i in indices.flatten()]
    logger.info(
        "apply_nms: %d → %d box sau NMS (score_thr=%.2f, nms_thr=%.2f)",
        len(boxes), len(result), score_threshold, nms_threshold,
    )
    return result
