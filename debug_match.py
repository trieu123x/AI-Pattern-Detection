"""
Debug script: reads the last uploaded pattern + drawing from .gradio/flagged
and prints diagnostic information about what the matcher is seeing.
"""
import cv2
import numpy as np
import os
import glob

from app.preprocess import preprocess_image, preprocess_pattern, crop_pattern

# ---------------------------------------------------------------------------
# 1. Find the latest flagged images (what the user uploaded to Gradio)
# ---------------------------------------------------------------------------
flagged_dir = ".gradio/flagged"
csv_path = os.path.join(flagged_dir, "dataset1.csv")

pattern_path = None
drawing_path = None

if os.path.exists(csv_path):
    with open(csv_path) as f:
        lines = f.read().strip().split("\n")
    if len(lines) > 1:
        last = lines[-1].split(",")
        if len(last) >= 2:
            pattern_path = last[0].strip()
            drawing_path = last[1].strip()

# Fallback: scan for any image files
if not pattern_path or not os.path.exists(pattern_path):
    imgs = glob.glob(os.path.join(flagged_dir, "**/*.png"), recursive=True) + \
           glob.glob(os.path.join(flagged_dir, "**/*.jpg"), recursive=True) + \
           glob.glob(os.path.join(flagged_dir, "**/*.webp"), recursive=True)
    imgs.sort(key=os.path.getmtime)
    if len(imgs) >= 2:
        pattern_path = imgs[-2]
        drawing_path = imgs[-1]

if not pattern_path or not drawing_path:
    print("[ERROR] Could not find flagged images. Please provide paths manually.")
    print("Usage: edit pattern_path and drawing_path at the top of this script.")
    # Manually set here if needed:
    # pattern_path = r"path\to\pattern.png"
    # drawing_path = r"path\to\drawing.png"
    exit(1)

print(f"Pattern : {pattern_path}")
print(f"Drawing : {drawing_path}")

# ---------------------------------------------------------------------------
# 2. Run the preprocessing pipeline
# ---------------------------------------------------------------------------
raw_pattern = cv2.imread(pattern_path)
raw_drawing = cv2.imread(drawing_path)

if raw_pattern is None:
    print(f"[ERROR] Could not read pattern image: {pattern_path}")
    exit(1)
if raw_drawing is None:
    print(f"[ERROR] Could not read drawing image: {drawing_path}")
    exit(1)

print(f"\nRaw pattern shape : {raw_pattern.shape}")
print(f"Raw drawing shape : {raw_drawing.shape}")

pattern = preprocess_pattern(raw_pattern)  # NO morphology for small pattern
drawing = preprocess_image(raw_drawing)
print(f"\nAfter preprocess:")
print(f"  pattern dtype={pattern.dtype}, min={pattern.min()}, max={pattern.max()}, shape={pattern.shape}")
print(f"  drawing dtype={drawing.dtype}, min={drawing.min()}, max={drawing.max()}, shape={drawing.shape}")

pattern_cropped = crop_pattern(pattern)
print(f"\nAfter crop_pattern:")
print(f"  pattern shape={pattern_cropped.shape}")

# ---------------------------------------------------------------------------
# 3. Try matchTemplate at scale=1.0, angle=0 and print top score
# ---------------------------------------------------------------------------
print("\n--- matchTemplate scores (scale=1.0, angle=0) ---")
if pattern_cropped.shape[0] > drawing.shape[0] or pattern_cropped.shape[1] > drawing.shape[1]:
    print("[WARNING] Pattern is LARGER than drawing! matchTemplate cannot run.")
    print(f"  pattern: {pattern_cropped.shape}  drawing: {drawing.shape}")
else:
    res = cv2.matchTemplate(drawing, pattern_cropped, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f"  Max score : {max_val:.4f}  at {max_loc}")
    print(f"  Min score : {min_val:.4f}")

    # Show distribution
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        count = int(np.sum(res >= thresh))
        print(f"  Pixels >= {thresh:.2f} : {count}")

# ---------------------------------------------------------------------------
# 4. Try multiple scales
# ---------------------------------------------------------------------------
print("\n--- Best score per scale ---")
for scale in [0.5, 0.6, 0.75, 1.0, 1.25, 1.5]:
    rp = cv2.resize(pattern_cropped, (0,0), fx=scale, fy=scale)
    if rp.shape[0] >= drawing.shape[0] or rp.shape[1] >= drawing.shape[1]:
        print(f"  scale={scale:.2f} -> SKIP (pattern too large: {rp.shape})")
        continue
    res = cv2.matchTemplate(drawing, rp, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    count_35 = int(np.sum(res >= 0.35))
    print(f"  scale={scale:.2f}  best={max_val:.4f}  at {max_loc}  (hits>=0.35: {count_35})")

# ---------------------------------------------------------------------------
# 5. Save debug images
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/debug_pattern_raw.png", pattern)
cv2.imwrite("outputs/debug_pattern_cropped.png", pattern_cropped)
cv2.imwrite("outputs/debug_drawing.png", drawing)
print("\n[OK] Debug images saved to outputs/")
