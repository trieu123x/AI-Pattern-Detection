from app.inference import run_inference

vis, boxes = run_inference(
    "pattern.png",
    "drawing.png"
)

print(boxes)
