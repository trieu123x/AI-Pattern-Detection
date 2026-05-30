# AI Pattern Detection

Hệ thống phát hiện ký hiệu kỹ thuật trong bản vẽ kỹ thuật số, hỗ trợ **xoay đa góc** và **thay đổi kích thước** tự động — không cần dữ liệu huấn luyện, chỉ cần 1 ảnh mẫu.

---

## Tính Năng

- **Zero-shot**: Chỉ cần 1 ảnh mẫu (template), không cần dataset hay training.
- **Multi-scale**: Tự động thử nhiều tỷ lệ kích thước (mặc định 8 mức).
- **Multi-angle**: Tự động thử nhiều góc xoay từ 0° đến 315° (mặc định bước 45°).
- **Auto-crop pattern**: Tự động loại bỏ viền trắng thừa của ảnh mẫu.
- **NMS hai lớp**: Lọc box chồng nhau chính xác ở cả cấp local và global.
- **Spotlight visualization**: Highlight vùng phát hiện, làm tối nền — dễ quan sát.
- **Giao diện Gradio**: Upload ảnh, điều chỉnh tham số và xem kết quả trực tiếp trên browser.

---

## Cài Đặt

### Yêu cầu

- Python ≥ 3.10
- pip

### Các bước

```bash
# 1. Clone hoặc tải project về
git clone <repo-url>
cd ai-pattern-detection

# 2. (Khuyến nghị) Tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

---

## Chạy Ứng Dụng

```bash
python gradio_app.py
```

Mở browser tại `http://localhost:7860` và:

1. **Upload Pattern** — ảnh mẫu ký hiệu cần tìm (PNG/JPG).
2. **Upload Drawing** — ảnh bản vẽ lớn (PNG/JPG).
3. Điều chỉnh **TM Threshold** (mặc định 0.70).
4. *(Tuỳ chọn)* Mở **Advanced Settings** để tuỳ chỉnh range của Scale và Angle.
5. Nhấn **Submit** → xem kết quả ảnh và JSON detections.

---

## Cấu Trúc Dự Án

```
ai-pattern-detection/
├── app/
│   ├── preprocess.py    # Tiền xử lý ảnh (grayscale, threshold, auto-crop)
│   ├── matcher.py       # Multi-scale × multi-angle template matching
│   ├── postprocess.py   # NMS toàn cục để gộp các box chồng nhau
│   ├── visualize.py     # Spotlight effect + bounding box + score label
│   ├── inference.py     # Orchestrator — điểm vào duy nhất của pipeline
│   └── encoder.py       # DINOv2 re-ranking (optional, tầng tăng precision)
├── examples/            # Ảnh mẫu thử nghiệm
├── outputs/             # Kết quả lưu ra
├── weights/             # Model weights (nếu dùng encoder.py)
├── gradio_app.py        # Giao diện Gradio
├── debug_match.py       # Script debug standalone
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Pattern Image ──► preprocess_pattern() ──► crop_pattern()
                                                  │
                                                  ▼
Drawing Image ──► preprocess_image() ──► match_pattern()
                                         (multi-scale × multi-angle
                                          TM_CCOEFF_NORMED + NMS nội bộ)
                                                  │
                                                  ▼
                                           apply_nms()   ← NMS toàn cục
                                                  │
                                                  ▼
                                          draw_boxes()   ← Spotlight effect
                                                  │
                                        ┌─────────┴──────────┐
                                        ▼                    ▼
                                  Output Image          JSON Detections
```

---

## Tham Số

### Threshold (`tm_threshold`)

| Giá trị | Ý nghĩa |
|---------|---------|
| 0.80 – 0.95 | Chính xác cao, ít false positive, có thể bỏ sót |
| **0.65 – 0.75** | **Cân bằng (khuyến nghị mặc định)** |
| 0.40 – 0.60 | Recall cao, nhiều candidate hơn |

### Advanced Settings

| Tham số | Mặc định | Ghi chú |
|---------|---------|---------|
| Scale min / max / step | 0.9 / 1.6 / 0.1 | 8 mức scale |
| Angle min / max / step | 0 / 315 / 45 | 8 góc xoay |

---

## Dependencies

| Thư viện | Vai trò |
|----------|---------|
| `opencv-python` | Template matching, NMS, image ops |
| `numpy` | Xử lý mảng |
| `Pillow` | Đọc/ghi ảnh |
| `gradio` | Giao diện web |
| `torch` + `timm` | DINOv2 encoder (optional) |
| `scikit-image` | Xử lý ảnh bổ sung |
| `scipy` | Tính toán khoa học |

---

## Ví Dụ Kết Quả

```json
{
  "total_detections": 3,
  "boxes": [
    {"bbox": [142, 305, 48, 32], "score": 0.8821, "angle": 0.0,   "scale": 1.0},
    {"bbox": [380, 210, 48, 32], "score": 0.8614, "angle": 0.0,   "scale": 1.1},
    {"bbox": [560, 410, 32, 48], "score": 0.8102, "angle": 90.0,  "scale": 1.0}
  ]
}
```

---

## Logging

Cấu hình log level trong code hoặc khi khởi chạy:

```python
import logging
logging.basicConfig(level=logging.DEBUG)   # Chi tiết (dev)
logging.basicConfig(level=logging.INFO)    # Tóm tắt   (production)
```

---

## License

MIT
