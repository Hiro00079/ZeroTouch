import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

class FaceMouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.margin = 20
        self.sensitivity = 0.6
        self.deadzone_camera = 8
        self.alpha = 0.12
        self.move_gate = 0.8

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)

        self.cursor_x = self.screen_w // 2
        self.cursor_y = self.screen_h // 2
        self.base_x = 0
        self.base_y = 0
        self.initialized = False

        self.filtered_dx = 0.0
        self.filtered_dy = 0.0
        self.zero_count = 0

        # ----- Blink detection (simple threshold with cooldown) -----
        self.left_ear_history = []      # to smooth or debug
        self.right_ear_history = []
        self.last_left_click = 0
        self.last_right_click = 0
        self.blink_cooldown = 0.8       # seconds between clicks
        
        # *** ADJUST THIS VALUE based on your test ***
        self.blink_threshold = 0.015    # change after running test_blink.py

        self.auto_recenter_start = time.time()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return frame, results

    def update(self, frame, results):
        if not results or not results.multi_face_landmarks:
            return frame

        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            # Nose for cursor
            nose = face_landmarks.landmark[168]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)

            if not self.initialized:
                self.base_x, self.base_y = nose_x, nose_y
                self.initialized = True

            raw_dx = nose_x - self.base_x
            raw_dy = nose_y - self.base_y

            if abs(raw_dx) < self.deadzone_camera:
                raw_dx = 0
            if abs(raw_dy) < self.deadzone_camera:
                raw_dy = 0

            if raw_dx == 0 and raw_dy == 0:
                self.zero_count += 1
                if self.zero_count >= 2:
                    self.filtered_dx, self.filtered_dy = 0.0, 0.0
            else:
                self.zero_count = 0
                self.filtered_dx = self.alpha * raw_dx + (1 - self.alpha) * self.filtered_dx
                self.filtered_dy = self.alpha * raw_dy + (1 - self.alpha) * self.filtered_dy

            move_mag = np.hypot(self.filtered_dx * self.sensitivity,
                                self.filtered_dy * self.sensitivity)
            if move_mag > self.move_gate:
                self.cursor_x += self.filtered_dx * self.sensitivity
                self.cursor_y += self.filtered_dy * self.sensitivity
                self.cursor_x = max(self.margin, min(self.screen_w - self.margin, self.cursor_x))
                self.cursor_y = max(self.margin, min(self.screen_h - self.margin, self.cursor_y))
                pyautogui.moveTo(self.cursor_x, self.cursor_y)

            if raw_dx == 0 and raw_dy == 0:
                if time.time() - self.auto_recenter_start > 1.0:
                    self.base_x, self.base_y = nose_x, nose_y
                    self.filtered_dx, self.filtered_dy = 0.0, 0.0
                    self.auto_recenter_start = time.time()
            else:
                self.auto_recenter_start = time.time()

            # ----- BLINK DETECTION (direct, without complex state) -----
            left_ear = face_landmarks.landmark[159].y - face_landmarks.landmark[145].y
            right_ear = face_landmarks.landmark[386].y - face_landmarks.landmark[374].y
            now = time.time()

            # Simple method: if ear goes below threshold, assume a blink just happened.
            # But to avoid multiple triggers per blink, we use a cooldown and a "blink ready" flag.
            # Here we trigger on the *falling edge*: when ear was above threshold and now below.
            # Store previous frame's values in class variables.
            if not hasattr(self, 'prev_left_ear'):
                self.prev_left_ear = left_ear
                self.prev_right_ear = right_ear

            # Left eye blink: transition from open (>= threshold) to closed (< threshold)
            if self.prev_left_ear >= self.blink_threshold and left_ear < self.blink_threshold:
                if (now - self.last_left_click) > self.blink_cooldown:
                    pyautogui.leftClick()
                    self.last_left_click = now
                    print(f"Left click! (EAR {left_ear:.4f})")
            # Right eye blink
            if self.prev_right_ear >= self.blink_threshold and right_ear < self.blink_threshold:
                if (now - self.last_right_click) > self.blink_cooldown:
                    pyautogui.rightClick()
                    self.last_right_click = now
                    print(f"Right click! (EAR {right_ear:.4f})")

            # Update previous values for next frame
            self.prev_left_ear = left_ear
            self.prev_right_ear = right_ear

            # Optional: print current EAR values occasionally (every 30 frames)
            if int(now * 30) % 30 == 0:
                print(f"L:{left_ear:.4f} R:{right_ear:.4f}", end="\r")

            # Draw nose
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

        return frame

    def release(self):
        self.cap.release()