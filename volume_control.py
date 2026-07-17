# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
from collections import deque

# ==============================
# CONFIGURATION
# ==============================
MIN_HAND_DISTANCE = 30
MAX_HAND_DISTANCE = 250
UPPER_THRESHOLD_PERCENT = 60
LOWER_THRESHOLD_PERCENT = 40
VOL_COOLDOWN_SECONDS = 0.15
PLAYPAUSE_COOLDOWN_SECONDS = 1.5

# ==============================
# INITIAL SETUP
# ==============================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils
webcam = cv2.VideoCapture(0)

last_vol_press_time = 0
last_playpause_time = 0
distance_buffer = deque(maxlen=5)  # Smoothing buffer

print("Hand Gesture Control Started! Press ESC to quit.")

# ==============================
# MAIN LOOP
# ==============================
while True:
    success, frame = webcam.read()
    if not success:
        print("⚠️ Webcam not detected.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)

    current_time = time.time()
    bar_color = (255, 0, 0)
    mode_text = "Idle"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            points = {
                i: (int(lm.x * w), int(lm.y * h))
                for i, lm in enumerate(hand_landmarks.landmark)
            }

            if 4 in points and 8 in points and 5 in points:
                x1, y1 = points[8]  # Index tip
                x2, y2 = points[4]  # Thumb tip
                y5 = points[5][1]   # Index knuckle
                y8 = points[8][1]   # Index tip (vertical check)

                distance = math.hypot(x2 - x1, y2 - y1)
                distance_buffer.append(distance)
                smooth_distance = np.mean(distance_buffer)

                volume_percent = np.interp(smooth_distance,
                                           [MIN_HAND_DISTANCE, MAX_HAND_DISTANCE],
                                           [0, 100])
                volume_percent = int(np.clip(volume_percent, 0, 100))

                # ----------- GESTURE: OPEN HAND (Volume) -----------
                if y8 < y5:
                    mode_text = "Volume Mode"
                    action = ""

                    if current_time - last_vol_press_time > VOL_COOLDOWN_SECONDS:
                        if volume_percent > UPPER_THRESHOLD_PERCENT:
                            pyautogui.press('volumeup')
                            bar_color = (0, 255, 0)
                            last_vol_press_time = current_time
                            action = "Volume Up"
                        elif volume_percent < LOWER_THRESHOLD_PERCENT:
                            pyautogui.press('volumedown')
                            bar_color = (0, 0, 255)
                            last_vol_press_time = current_time
                            action = "Volume Down"

                    # Volume Bar
                    cv2.rectangle(frame, (w - 85, 100), (w - 50, 400), (255, 255, 255), 2)
                    bar_height = int(np.interp(volume_percent, [0, 100], [400, 100]))
                    cv2.rectangle(frame, (w - 85, bar_height), (w - 50, 400), bar_color, -1)
                    cv2.putText(frame, f'{volume_percent}%', (w - 120, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if action:
                        cv2.putText(frame, action, (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                # ----------- GESTURE: CLOSED FIST (Play/Pause) -----------
                elif y8 > y5:
                    mode_text = "Play/Pause"
                    if current_time - last_playpause_time > PLAYPAUSE_COOLDOWN_SECONDS:
                        pyautogui.press('playpause')
                        last_playpause_time = current_time
                        cv2.putText(frame, "Toggled Media", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # ==============================
    # UI ELEMENTS
    # ==============================
    cv2.putText(frame, f"Mode: {mode_text}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Media Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# ==============================
# CLEANUP
# ==============================
webcam.release()
cv2.destroyAllWindows()
print("Program ended successfully.")
