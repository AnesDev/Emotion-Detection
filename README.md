Emotion-Detection is an interactive app that uses AI to recognize human emotions in real time from a camera feed or images.

The system works in two steps: first, it detects faces using OpenCV, identifying where people are in the frame. Then, each detected face is analyzed by a deep learning model (TensorFlow) that predicts emotions like 😀 Happy, 😢 Sad, 😮 Surprised, or 😠 Angry.

For a smooth, engaging experience, the demo overlays an emoji and emotion label on each face instantly, updating as expressions change. It also includes smart smoothing to make predictions more stable and accurate.

Built with Python, OpenCV, TensorFlow, and Pillow, the demo is ready to run—no coding needed. Launch it, and see AI interpret emotions in real time.