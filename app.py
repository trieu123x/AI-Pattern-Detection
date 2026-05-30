# Entry point for Hugging Face Spaces (must be named app.py)
# All logic lives in gradio_app.py
from gradio_app import demo

if __name__ == "__main__":
    demo.launch()
