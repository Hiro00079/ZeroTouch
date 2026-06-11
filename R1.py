import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
screen_w, screen_h = pyautogui.size()
margin = 20

# Control mode: "absolute" or "velocity"
MODE = "absolute"          # default, can be toggled with 'm'

# Smoothing: "kalman" or "ema"
FILTER_TYPE = "kalman"

# EMA smoothing factor (only used if FILTER_TYPE == "ema")
EMA_ALPHA = 0.15

# Velocity mode parameters
VEL_GAIN = 0.8             # displacement → speed (pixels of cursor per frame per pixel of head movement)
VEL_DEADZONE = 5           # ignore tiny head displacement
MAX_VEL = 20               # maximum cursor movement per frame (pixels)

# Absolute mode parameters
ABS_SMOOTH = 0.12          # additional EMA for absolute mapping (if not using Kalman)

# Gesture thresholds
EYEBROW_RAISE_THRESH = -0.03   # negative when raised (y difference brow - eye centre)
EYEBROW_FURROW_THRESH = 0.02   # positive when inner brow below eye centre
WINK_EAR_THRESH = 0.15         # Eye Aspect Ratio below this = closed
OPEN_EYE_MIN_EAR = 0.20        # Eye considered open above this
HEAD_TILT_DEADZONE = 8.0       # degrees
HSCROLL_GAIN = 2               # pixels per degree tilt
GESTURE_COOLDOWN = 0.15        # seconds between same gesture actions
CLICK_COOLDOWN = 1.0           # seconds between clicks

# -------------------------------
# Kalman filter (1D, constant velocity model)
# -------------------------------
class KalmanFilter1D:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, initial_estimate=0.0):
        self.x = initial_estimate      # state (position)
        self.p = 1.0                   # estimation error covariance
        self.q = process_noise         # process noise
        self.r = measurement_noise     # measurement noise

    def update(self, measurement):
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        return self.x

# -------------------------------
# Gesture helper functions
# -------------------------------
def get_eye_aspect_ratio(landmarks, left=True):
    """EAR for specified eye."""
    if left:
        top = landmarks.landmark[159]
        bottom = landmarks.landmark[145]
        left_corner = landmarks.landmark[133]
        right_corner = landmarks.landmark[33]
    else:
        top = landmarks.landmark[386]
        bottom = landmarks.landmark[374]
        left_corner = landmarks.landmark[362]
        right_corner = landmarks.landmark[263]
    vert_dist = abs(top.y - bottom.y)
    horiz_dist = abs(left_corner.x - right_corner.x)
    return vert_dist / horiz_dist if horiz_dist > 0 else 0

def get_eyebrow_raise(landmarks, left=True):
    """Vertical distance from eyebrow outer to eye centre (normalised)."""
    if left:
        brow = landmarks.landmark[70]      # left eyebrow outer
        eye_centre = landmarks.landmark[33]  # left eye centre
    else:
        brow = landmarks.landmark[300]     # right eyebrow outer
        eye_centre = landmarks.landmark[263] # right eye centre
    return brow.y - eye_centre.y   # negative when raised

def get_head_tilt(landmarks):
    """Angle of line between left and right eye centres (degrees)."""
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = np.degrees(np.arctan2(dy, dx))
    return angle

# -------------------------------
# Initialize webcam and MediaPipe
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

# Cursor state
cursor_x, cursor_y = screen_w // 2, screen_h // 2

# For absolute mode: smoothed nose position
smoothed_nose_x, smoothed_nose_y = 0.0, 0.0

# For velocity mode: neutral reference point
neutral_x, neutral_y = 0, 0
neutral_initialized = False

# Kalman filters (one per coordinate)
kf_x = KalmanFilter1D()
kf_y = KalmanFilter1D()

# Gesture timing variables
last_gesture_time = {
    'scroll_up': 0,
    'scroll_down': 0,
    'left_wink': 0,
    'right_wink': 0,
    'hscroll': 0
}
last_click_time = 0
auto_recenter_start = time.time()

# Calibration storage (for eyebrow neutral reference - optional)
calibrated_neutral_eyebrow = None

# -------------------------------
# Helper: apply smoothing to raw landmark
# -------------------------------
def smooth_point(raw_x, raw_y, filter_type, ema_alpha, prev_x, prev_y):
    if filter_type == "kalman":
        smooth_x = kf_x.update(raw_x)
        smooth_y = kf_y.update(raw_y)
        return smooth_x, smooth_y
    else:  # EMA
        smooth_x = ema_alpha * raw_x + (1 - ema_alpha) * prev_x
        smooth_y = ema_alpha * raw_y + (1 - ema_alpha) * prev_y
        return smooth_x, smooth_y

# -------------------------------
# Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Nose bridge (landmark 168) for cursor movement
            nose = face_landmarks.landmark[168]
            raw_nose_x = nose.x * w
            raw_nose_y = nose.y * h

            # Apply smoothing to the raw nose position
            if FILTER_TYPE == "kalman":
                curr_nose_x = kf_x.update(raw_nose_x)
                curr_nose_y = kf_y.update(raw_nose_y)
            else:  # EMA
                if 'prev_nose_x' not in dir():
                    prev_nose_x, prev_nose_y = raw_nose_x, raw_nose_y
                curr_nose_x = EMA_ALPHA * raw_nose_x + (1 - EMA_ALPHA) * prev_nose_x
                curr_nose_y = EMA_ALPHA * raw_nose_y + (1 - EMA_ALPHA) * prev_nose_y
                prev_nose_x, prev_nose_y = curr_nose_x, curr_nose_y

            # -------------------------------
            # Mode 1: Absolute mapping
            # -------------------------------
            if MODE == "absolute":
                abs_x = np.interp(curr_nose_x, [0, w], [margin, screen_w - margin])
                abs_y = np.interp(curr_nose_y, [0, h], [margin, screen_h - margin])
                if 'last_abs_x' not in dir():
                    last_abs_x, last_abs_y = abs_x, abs_y
                abs_x = EMA_ALPHA * abs_x + (1 - EMA_ALPHA) * last_abs_x
                abs_y = EMA_ALPHA * abs_y + (1 - EMA_ALPHA) * last_abs_y
                last_abs_x, last_abs_y = abs_x, abs_y

                cursor_x = np.clip(abs_x, margin, screen_w - margin)
                cursor_y = np.clip(abs_y, margin, screen_h - margin)
                pyautogui.moveTo(cursor_x, cursor_y)

            # -------------------------------
            # Mode 2: Relative velocity control
            # -------------------------------
            else:  # velocity mode
                if not neutral_initialized:
                    neutral_x, neutral_y = curr_nose_x, curr_nose_y
                    neutral_initialized = True

                dx_raw = curr_nose_x - neutral_x
                dy_raw = curr_nose_y - neutral_y

                if abs(dx_raw) < VEL_DEADZONE:
                    dx_raw = 0
                if abs(dy_raw) < VEL_DEADZONE:
                    dy_raw = 0

                vel_x = dx_raw * VEL_GAIN
                vel_y = dy_raw * VEL_GAIN
                vel_x = np.clip(vel_x, -MAX_VEL, MAX_VEL)
                vel_y = np.clip(vel_y, -MAX_VEL, MAX_VEL)

                cursor_x += vel_x
                cursor_y += vel_y
                cursor_x = np.clip(cursor_x, margin, screen_w - margin)
                cursor_y = np.clip(cursor_y, margin, screen_h - margin)
                pyautogui.moveTo(cursor_x, cursor_y)

                if dx_raw == 0 and dy_raw == 0:
                    if time.time() - auto_recenter_start > 1.0:
                        neutral_x, neutral_y = curr_nose_x, curr_nose_y
                        auto_recenter_start = time.time()
                else:
                    auto_recenter_start = time.time()

            # -------------------------------
            # NEW GESTURES (1, 3, 4, 6)
            # -------------------------------
            now = time.time()

            # ---- Gesture 1: Eyebrow Raise -> Scroll Up ----
            brow_l = get_eyebrow_raise(face_landmarks, left=True)
            brow_r = get_eyebrow_raise(face_landmarks, left=False)
            # Both eyebrows raised above threshold
            if brow_l < EYEBROW_RAISE_THRESH and brow_r < EYEBROW_RAISE_THRESH:
                if now - last_gesture_time['scroll_up'] > GESTURE_COOLDOWN:
                    pyautogui.scroll(3)   # scroll up
                    last_gesture_time['scroll_up'] = now

            # ---- Gesture 1 also: Frown (optional) but we only use raise for up, you could add down with furrow
            # Actually for scroll down we can use eyebrow furrow (inner brows down). Add that as a bonus:
            inner_brow_l = face_landmarks.landmark[65].y
            inner_brow_r = face_landmarks.landmark[295].y
            eye_centre_y = (face_landmarks.landmark[33].y + face_landmarks.landmark[263].y) / 2
            if inner_brow_l > eye_centre_y + EYEBROW_FURROW_THRESH and inner_brow_r > eye_centre_y + EYEBROW_FURROW_THRESH:
                if now - last_gesture_time['scroll_down'] > GESTURE_COOLDOWN:
                    pyautogui.scroll(-3)  # scroll down
                    last_gesture_time['scroll_down'] = now

            # ---- Gesture 3: Left Wink (left eye closed, right open) -> Left Click ----
            left_ear = get_eye_aspect_ratio(face_landmarks, left=True)
            right_ear = get_eye_aspect_ratio(face_landmarks, left=False)
            if left_ear < WINK_EAR_THRESH and right_ear > OPEN_EYE_MIN_EAR:
                if now - last_gesture_time['left_wink'] > CLICK_COOLDOWN:
                    pyautogui.leftClick()
                    last_gesture_time['left_wink'] = now
                    last_click_time = now

            # ---- Gesture 4: Right Wink (right eye closed, left open) -> Right Click ----
            if right_ear < WINK_EAR_THRESH and left_ear > OPEN_EYE_MIN_EAR:
                if now - last_gesture_time['right_wink'] > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_gesture_time['right_wink'] = now
                    last_click_time = now

            # ---- Gesture 6: Head Tilt -> Horizontal Scroll ----
            tilt_angle = get_head_tilt(face_landmarks)
            if abs(tilt_angle) > HEAD_TILT_DEADZONE:
                # Scroll amount proportional to tilt beyond deadzone
                scroll_h = int((tilt_angle - HEAD_TILT_DEADZONE) / HSCROLL_GAIN) if tilt_angle > 0 else int((tilt_angle + HEAD_TILT_DEADZONE) / HSCROLL_GAIN)
                if abs(scroll_h) > 0 and (now - last_gesture_time['hscroll'] > GESTURE_COOLDOWN):
                    # Note: pyautogui.hscroll may not exist on all platforms; fallback to key simulation or ignore.
                    # Here we use pyautogui.hscroll if available, else simulate left/right arrow keys.
                    try:
                        pyautogui.hscroll(scroll_h)
                    except AttributeError:
                        # Fallback: send left/right arrow keys
                        if scroll_h > 0:
                            pyautogui.press('right')
                        elif scroll_h < 0:
                            pyautogui.press('left')
                    last_gesture_time['hscroll'] = now

            # Draw nose point for visual feedback
            nose_pixel = (int(curr_nose_x), int(curr_nose_y))
            cv2.circle(frame, nose_pixel, 5, (0, 0, 255), -1)

            # Optional: display gesture status
            cv2.putText(frame, f"Brow: {brow_l:.3f} {brow_r:.3f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"Tilt: {tilt_angle:.1f} deg", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Display mode and instructions
    cv2.putText(frame, f"Mode: {MODE.upper()}   (press 'm' to toggle)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Filter: " + FILTER_TYPE.upper(), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'r' to reset neutral (vel) / recenter (abs)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "Gestures: Raise brows=scroll up | Furrow=scroll down | Left wink=left click | Right wink=right click | Head tilt=horizontal scroll", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    small_frame = cv2.resize(frame, (400, 300))
    cv2.imshow("Advanced Face Mouse", small_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 13:   # ESC or Enter
        break
    elif key == ord('m'):
        MODE = "velocity" if MODE == "absolute" else "absolute"
        if MODE == "velocity":
            neutral_initialized = False
        else:
            if 'last_abs_x' in dir():
                del last_abs_x
        print(f"Switched to {MODE} mode")
    elif key == ord('r'):
        if MODE == "velocity":
            neutral_x, neutral_y = curr_nose_x, curr_nose_y
            print("Neutral point reset")
        else:
            if results.multi_face_landmarks:
                cursor_x = np.interp(curr_nose_x, [0, w], [margin, screen_w - margin])
                cursor_y = np.interp(curr_nose_y, [0, h], [margin, screen_h - margin])
                pyautogui.moveTo(cursor_x, cursor_y)
                print("Cursor recentered")

cap.release()
cv2.destroyAllWindows()