from controller import FaceMouseController
from gui import GUI
from voice import VoiceTyper
import cv2

def main():
    controller = FaceMouseController()
    gui = GUI()
    voice = VoiceTyper(device_index=2)  # <-- ADD THIS (use index 2)
    voice.start()

    print("Face Mouse Started")
    print("Controls:")
    print("  ESC or ENTER → Exit")
    print("  R → Manual recalibration")
    print("  Voice: Say 'type' followed by your sentence → types the sentence")

    while True:
        frame, results = controller.get_frame()
        if frame is None:
            break

        frame = controller.update(frame, results)
        status = "Voice: listening for 'type'"
        gui.show(frame, status)

        key = gui.get_key(1)
        if key in (27, 13):
            break
        elif key == ord('r'):
            controller.initialized = False
            print("Manual recalibration")

    voice.stop()
    controller.release()
    gui.destroy()
    print("Exited")

if __name__ == "__main__":
    main()