import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont

# Emotion to emoji mapping
EMOTION_EMOJIS = {
    'neutral': '😐',
    'happiness': '😊',
    'surprise': '😮',
    'sadness': '😢',
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'contempt': '😒'
}

# Color scheme for each emotion
EMOTION_COLORS = {
    'neutral': (200, 200, 200),
    'happiness': (0, 255, 100),
    'surprise': (255, 200, 0),
    'sadness': (255, 100, 100),
    'anger': (0, 0, 255),
    'disgust': (150, 255, 0),
    'fear': (200, 0, 200),
    'contempt': (100, 150, 200)
}

EMOTION_LABELS = list(EMOTION_EMOJIS.keys())

class EmojiEmotionDetector:
    """Clean, fast emotion detection with emojis - Club Demo Edition"""
    
    def __init__(self, model_path):
        print("🔄 Loading AI model...")
        self.model = load_model(model_path, compile=False)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load emoji fonts
        self.emoji_fonts = []
        font_sizes = [96, 72, 48]  # Large, medium, small
        for size in font_sizes:
            try:
                font = ImageFont.truetype("seguiemj.ttf", size)
                self.emoji_fonts.append(font)
            except:
                try:
                    font = ImageFont.truetype("AppleColorEmoji.ttf", size)
                    self.emoji_fonts.append(font)
                except:
                    try:
                        font = ImageFont.truetype("NotoColorEmoji.ttf", size)
                        self.emoji_fonts.append(font)
                    except:
                        font = ImageFont.load_default()
                        self.emoji_fonts.append(font)
        
        # Performance optimizations
        self.prediction_history = deque(maxlen=10)  # Much longer for stability
        # EXTREME rebalancing - make rare emotions actually appear
        self.weights = np.array([
            0.15,  # neutral - crushed
            0.15,  # happiness - crushed
            4.0,   # surprise - EXTREME boost
            2.5,   # sadness - strong boost
            0.8,   # anger - nerfed hard (was stealing from surprise)
            4.5,   # disgust - EXTREME boost
            4.2,   # fear - EXTREME boost
            3.8    # contempt - EXTREME boost
        ])
        self.frame_skip = 1
        self.frame_count = 0
        self.last_emotions = {}
        self.confidence_threshold = 0.3  # Higher threshold for stability
        
        print("✅ Ready!")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset stats")
        print("  'f' - Fullscreen (for expo!)")
        print("  'w' - Window mode\n")
        
    def preprocess_face(self, face_img):
        """Enhanced face preprocessing for better detection"""
        face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_CUBIC)
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Stronger contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        face_img = clahe.apply(face_img)
        
        # Additional histogram equalization
        face_img = cv2.equalizeHist(face_img)
        
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])
        return face_img
    
    def get_emotion(self, face_roi, face_id):
        """Enhanced emotion detection with better accuracy"""
        if self.frame_count % self.frame_skip == 0:
            processed = self.preprocess_face(face_roi)
            preds = self.model.predict(processed, verbose=0)[0]
            
            # Softmax normalization
            if np.max(preds) > 1.0:
                preds = np.exp(preds - np.max(preds))
                preds = preds / preds.sum()
            
            # Apply stronger weights
            preds = preds * self.weights
            preds = preds / preds.sum()
            
            # Temporal smoothing with weighted average (favor recent)
            self.prediction_history.append(preds)
            if len(self.prediction_history) > 1:
                weights = np.linspace(0.5, 1.0, len(self.prediction_history))
                weights = weights / weights.sum()
                smoothed = np.average(self.prediction_history, axis=0, weights=weights)
            else:
                smoothed = preds
            
            emotion_idx = np.argmax(smoothed)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = smoothed[emotion_idx]
            
            # Only update if confidence is above threshold
            if confidence > self.confidence_threshold:
                self.last_emotions[face_id] = (emotion, EMOTION_EMOJIS[emotion], confidence)
        
        return self.last_emotions.get(face_id, ('neutral', '😐', 0.5))
    
    def draw_emoji(self, frame, emoji, x, y, size_index=0):
        """Draw emoji using PIL"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = self.emoji_fonts[min(size_index, len(self.emoji_fonts) - 1)]
        
        # Shadow for depth
        draw.text((x + 3, y + 3), emoji, font=font, fill=(0, 0, 0, 128))
        draw.text((x, y), emoji, font=font, embedded_color=True)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def draw_text_with_outline(self, frame, text, position, font, scale, color, thickness):
        """Draw text with black outline for better visibility"""
        x, y = position
        # Outline
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
        # Main text
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)
    
    def run(self):
        """Run the detector"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_time = time.time()
        fps = 0
        
        # Animation variables
        pulse = 0
        
        print("🎥 Camera active! Show your emotions!\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            pulse += 0.1
            frame = cv2.flip(frame, 1)
            
            # Detect faces with better parameters
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply slight blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.15,  # Less sensitive to prevent duplicates
                minNeighbors=6,    # Higher to prevent duplicate detections
                minSize=(100, 100),  # Larger minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Remove overlapping faces (keep only the best one)
            if len(faces) > 1:
                # Sort by area (largest first)
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                filtered_faces = []
                for face in faces:
                    x, y, w, h = face
                    # Check if this face overlaps with any already accepted face
                    is_duplicate = False
                    for accepted in filtered_faces:
                        ax, ay, aw, ah = accepted
                        # Calculate overlap
                        x_overlap = max(0, min(x + w, ax + aw) - max(x, ax))
                        y_overlap = max(0, min(y + h, ay + ah) - max(y, ay))
                        overlap_area = x_overlap * y_overlap
                        face_area = w * h
                        # If overlap is more than 30%, consider it duplicate
                        if overlap_area > 0.3 * face_area:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        filtered_faces.append(face)
                faces = filtered_faces
            
            # Process faces
            for idx, (x, y, w, h) in enumerate(faces):
                face_roi = gray[y:y+h, x:x+w]
                
                try:
                    emotion, emoji, confidence = self.get_emotion(face_roi, idx)
                    color = EMOTION_COLORS[emotion]
                    
                    # Simple clean box - no glow
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw BIG emoji above face
                    emoji_size = max(80, w // 3)
                    emoji_x = x + w // 2 - emoji_size // 2
                    emoji_y = max(20, y - emoji_size - 20)
                    
                    frame = self.draw_emoji(frame, emoji, emoji_x, emoji_y, size_index=0)
                    
                    # Emotion label below face with styled background
                    label = f"{emotion.upper()}"
                    label_y = y + h + 40
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
                    
                    # Rounded background effect
                    padding = 15
                    cv2.ellipse(frame, (x + tw // 2 + padding, label_y - th // 2 - 5),
                               (tw // 2 + padding, th // 2 + padding), 0, 0, 360, 
                               (0, 0, 0), -1)
                    
                    # Draw label
                    self.draw_text_with_outline(frame, label, (x + padding, label_y),
                                               cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
                
                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")
            
            # Calculate FPS
            fps = 1.0 / (time.time() - fps_time + 0.001)
            fps_time = time.time()
            
            # Clean minimal UI - just FPS
            self.draw_text_with_outline(frame, f"FPS: {int(fps)}", (20, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Tip text at bottom
            if len(faces) == 0:
                tip = "Position your face in the frame"
                (tw, th), _ = cv2.getTextSize(tip, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tip_x = (frame.shape[1] - tw) // 2
                self.draw_text_with_outline(frame, tip, (tip_x, frame.shape[0] - 40),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Emotion Detector', frame)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r'):
                self.prediction_history.clear()
                self.last_emotions.clear()
                print("♻️ Reset!")
            elif key == ord('f'):  # Fullscreen
                cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_FULLSCREEN)
            elif key == ord('w'):  # Window mode
                cv2.setWindowProperty('Emotion Detector', cv2.WND_PROP_FULLSCREEN, 
                                     cv2.WINDOW_NORMAL)
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✨ Thanks for trying the Emotion Detector!")


if __name__ == "__main__":
    MODEL_PATH = "ferplus_model_mv_best.h5"
    
    try:
        detector = EmojiEmotionDetector(MODEL_PATH)
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()