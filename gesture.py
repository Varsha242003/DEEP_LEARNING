"""
Hand Gesture Control Application
==================================
Modules:
  1. Webcam Input & Preprocessing
  2. Gesture Interception (ROI detection)
  3. Gesture Recognition (Keras model)
  4. Command Execution (keyboard/system actions)

Requirements:
    pip install opencv-python tensorflow numpy pyautogui
"""

import cv2
import numpy as np
import tensorflow as tf
import pyautogui
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH = "gesture_model.keras"   
ROI_TOP, ROI_RIGHT, ROI_BOTTOM, ROI_LEFT = 100, 350, 400, 650  # Region of Interest box

# 25 ASL classes (A-Z minus J, since J needs motion)
GESTURE_LABELS = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S',
    'T','U','V','W','X','Y','Z'
]

# ─────────────────────────────────────────────
# MODULE 4: COMMAND MAPPING
# Map each gesture label to a computer action
# ─────────────────────────────────────────────
def execute_command(gesture: str):
    """
    Module 4 – Execute a system command based on the detected gesture.
    Customize this dictionary to change what each gesture does.
    """
    commands = {
        'V': lambda: pyautogui.hotkey('ctrl', 'v'),          # Paste
        'C': lambda: pyautogui.hotkey('ctrl', 'c'),          # Copy
        'Z': lambda: pyautogui.hotkey('ctrl', 'z'),          # Undo
        'S': lambda: pyautogui.hotkey('ctrl', 's'),          # Save
        'N': lambda: pyautogui.hotkey('ctrl', 'n'),          # New
        'P': lambda: pyautogui.press('playpause'),           # Play/Pause media
        'U': lambda: pyautogui.press('volumeup'),            # Volume Up
        'D': lambda: pyautogui.press('volumedown'),          # Volume Down
        'M': lambda: pyautogui.press('volumemute'),          # Mute
        'L': lambda: pyautogui.press('nexttrack'),           # Next track
        'B': lambda: pyautogui.press('prevtrack'),           # Previous track
        'R': lambda: pyautogui.hotkey('alt', 'F4'),          # Close window
        'T': lambda: pyautogui.hotkey('win', 't'),           # Taskbar
        'W': lambda: pyautogui.hotkey('win', 'up'),          # Maximize window
        'X': lambda: pyautogui.hotkey('win', 'd'),           # Show desktop
        'F': lambda: pyautogui.press('f11'),                 # Fullscreen
        'O': lambda: pyautogui.hotkey('win', 'o'),           # Rotation lock
        'A': lambda: pyautogui.hotkey('ctrl', 'a'),          # Select all
        'E': lambda: pyautogui.hotkey('win', 'e'),           # File explorer
        'I': lambda: pyautogui.hotkey('ctrl', 'shift', 'i'),# Dev tools
        'G': lambda: pyautogui.hotkey('win', 'g'),           # Game bar
        'H': lambda: pyautogui.hotkey('win', 'h'),           # Dictation
        'K': lambda: pyautogui.hotkey('win', 'k'),           # Connect
        'Q': lambda: pyautogui.hotkey('alt', 'tab'),         # Switch window
        'Y': lambda: pyautogui.hotkey('ctrl', 'y'),          # Redo
    }

    action = commands.get(gesture)
    if action:
        try:
            action()
            return True
        except Exception as e:
            print(f"[CMD] Error executing command for '{gesture}': {e}")
    return False


# ─────────────────────────────────────────────
# MODULE 1: WEBCAM INPUT & PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_frame(roi_frame: np.ndarray) -> np.ndarray:
    """
    Module 1 – Convert ROI frame into model-ready input.
    Steps: grayscale → resize to 28x28 → normalize → reshape
    """
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    model_input = normalized.reshape(1, 28, 28, 1)   # batch of 1
    return model_input


# ─────────────────────────────────────────────
# MODULE 2: GESTURE INTERCEPTION (ROI)
# ─────────────────────────────────────────────
def extract_roi(frame: np.ndarray) -> np.ndarray:
    """
    Module 2 – Crop the Region of Interest where the hand should be placed.
    Also applies background subtraction to isolate the hand silhouette.
    """
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]
    return roi


def draw_roi_box(frame: np.ndarray) -> np.ndarray:
    """Draw the ROI rectangle guide on screen."""
    cv2.rectangle(frame, (ROI_RIGHT, ROI_TOP), (ROI_LEFT, ROI_BOTTOM),
                  (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (ROI_RIGHT, ROI_TOP - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


# ─────────────────────────────────────────────
# MODULE 3: GESTURE RECOGNITION
# ─────────────────────────────────────────────
class GestureRecognizer:
    """Module 3 – Load model and predict gesture from preprocessed frame."""

    def __init__(self, model_path: str):
        print(f"[MODEL] Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.labels = GESTURE_LABELS
        print(f"[MODEL] Ready. Classes: {self.labels}")

    def predict(self, model_input: np.ndarray, threshold: float = 0.7):
        """
        Returns (gesture_label, confidence) if confidence > threshold,
        else returns (None, confidence).
        """
        predictions = self.model.predict(model_input, verbose=0)[0]
        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))

        if confidence >= threshold:
            return self.labels[class_idx], confidence
        return None, confidence


# ─────────────────────────────────────────────
# MAIN APPLICATION LOOP
# ─────────────────────────────────────────────
def main():
    # Init
    recognizer = GestureRecognizer(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        return

    print("\n[APP] Gesture Control App Running")
    print("      Press 'q' to quit | Press 'p' to pause/resume commands\n")

    # State
    last_gesture = None
    last_command_time = 0
    command_cooldown = 1.5   # seconds between commands (prevent spam)
    commands_paused = False
    gesture_hold_frames = 0
    hold_threshold = 8       # frames gesture must be stable before firing

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)   # Mirror for natural feel

        # ── Module 2: Extract ROI ──
        draw_roi_box(frame)
        roi = extract_roi(frame)

        # ── Module 1: Preprocess ──
        model_input = preprocess_frame(roi)

        # ── Module 3: Recognize ──
        gesture, confidence = recognizer.predict(model_input, threshold=0.75)

        # Track gesture stability
        if gesture == last_gesture:
            gesture_hold_frames += 1
        else:
            gesture_hold_frames = 0
            last_gesture = gesture

        # ── Module 4: Execute command ──
        now = time.time()
        command_fired = False

        if (gesture is not None
                and gesture_hold_frames == hold_threshold   # fire once on stable hold
                and not commands_paused
                and (now - last_command_time) > command_cooldown):

            command_fired = execute_command(gesture)
            if command_fired:
                last_command_time = now
                print(f"[CMD] Gesture '{gesture}' ({confidence:.0%}) → command executed")

        # ── Display overlay ──
        # Show ROI preview (small, top-right corner)
        roi_display = cv2.resize(roi, (100, 100))
        frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = roi_display

        # Gesture label
        label_text = f"Gesture: {gesture or '---'} ({confidence:.0%})"
        color = (0, 255, 0) if gesture else (0, 100, 255)
        cv2.putText(frame, label_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Stability bar
        bar_width = int((gesture_hold_frames / hold_threshold) * 200)
        cv2.rectangle(frame, (10, 60), (210, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 60), (10 + min(bar_width, 200), 80), (0, 200, 255), -1)
        cv2.putText(frame, "Stability", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Command fired flash
        if command_fired or (now - last_command_time) < 0.4:
            cv2.putText(frame, f"COMMAND: {last_gesture}!", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Paused indicator
        if commands_paused:
            cv2.putText(frame, "COMMANDS PAUSED (press P)", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Controls hint
        cv2.putText(frame, "Q: Quit | P: Pause commands", (10, frame.shape[0] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow("Gesture Control App", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[APP] Quitting...")
            break
        elif key == ord('p'):
            commands_paused = not commands_paused
            state = "PAUSED" if commands_paused else "ACTIVE"
            print(f"[APP] Commands {state}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()