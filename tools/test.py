import speech_recognition as sr

def test_mic(index):
    r = sr.Recognizer()
    try:
        with sr.Microphone(device_index=index) as source:
            print(f"Testing index {index}: {sr.Microphone.list_microphone_names()[index]}")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Say something like 'hello'...")
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            text = r.recognize_google(audio)
            print(f"✓ SUCCESS! Heard: {text}\n")
            return True
    except sr.WaitTimeoutError:
        print("✗ No speech detected (timeout)\n")
    except sr.UnknownValueError:
        print("✗ Heard something but could not understand\n")
    except Exception as e:
        print(f"✗ Error: {e}\n")
    return False

# List all microphones
print("Available microphones:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"  {i}: {name}")

# Test indices that are likely input devices
candidates = [1, 2, 8, 14, 15, 20, 22, 32, 33]
working = []
for idx in candidates:
    if test_mic(idx):
        working.append(idx)

print(f"Working mic indices: {working}")