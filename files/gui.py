import cv2

class GUI:
    def __init__(self, window_name="Face Mouse", window_size=(400, 300)):
        self.window_name = window_name
        self.window_size = window_size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.window_size)

    def show(self, frame, status_text=None):
        if frame is None:
            return
        small_frame = cv2.resize(frame, self.window_size)
        if status_text:
            cv2.putText(small_frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.window_name, small_frame)

    def get_key(self, timeout=1):
        return cv2.waitKey(timeout)

    def destroy(self):
        cv2.destroyAllWindows()