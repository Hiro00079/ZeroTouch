import speech_recognition as sr
import pyautogui
import threading
import time
import platform

class VoiceTyper:
    def __init__(self, device_index=None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=device_index)
        self.running = True

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _type_text(self, text):
        try:
            pyautogui.write(text)
            print(f"✓ Typed: {text}")
            return True
        except Exception as e:
            print(f"pyautogui.write failed: {e}")
        try:
            import pyperclip
            pyperclip.copy(text)
            time.sleep(0.1)
            if platform.system() == "Darwin":
                pyautogui.hotkey('command', 'v')
            else:
                pyautogui.hotkey('ctrl', 'v')
            print(f"✓ Pasted: {text}")
            return True
        except ImportError:
            print("pyperclip not installed. Run: pip install pyperclip")
        except Exception as e:
            print(f"Clipboard paste failed: {e}")
        return False

    def _listen_loop(self):
        with self.microphone as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.recognizer.energy_threshold = 2000
            self.recognizer.dynamic_energy_threshold = True
            print("\nVoice Typer Ready.")
            print("Say: 'type' followed by your sentence (e.g., 'type hello world')\n")

            while self.running:
                try:
                    print(" Listening...")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=6)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f" Heard: \"{text}\"")

                    if "stop listening" in text:
                        print(" Stopping...")
                        self.running = False
                        break

                    if "type" in text:
                        parts = text.split("type", 1)
                        if len(parts) > 1:
                            text_to_type = parts[1].strip()
                            if text_to_type:
                                print(f" Extracted: \"{text_to_type}\"")
                                print(" Click on the target window (Notepad, browser, etc.)")
                                for i in range(5, 0, -1):
                                    print(f"   Typing in {i} seconds...", end="\r")
                                    time.sleep(1)
                                print("                              \n")
                                self._type_text(text_to_type)
                                time.sleep(1)
                            else:
                                print(" No text after 'type'")
                        else:
                            print(" Could not extract text")
                    else:
                        # Remind user to say "type"
                        print(" You didn't say 'type'. Try: 'type hello world'")

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print(" Could not understand – please speak clearly")
                except sr.RequestError as e:
                    print(f" API error: {e}")
                except Exception as e:
                    print(f" Error: {e}")

if __name__ == "__main__":
    print("Available microphones:")
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  {i}: {name}")
    try:
        idx = int(input("\nEnter microphone index (or press Enter for default): ") or -1)
        typer = VoiceTyper(device_index=idx if idx >= 0 else None)
    except:
        typer = VoiceTyper()
    typer.start()
    print("\nVoice typer running. Say 'stop listening' or Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.stop()
        print("\nStopped.")