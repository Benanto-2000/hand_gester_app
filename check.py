import sys
import os

print("--- DIAGNOSTIC CHECK ---")
print("Python Executable:", sys.executable)
print("Current Directory Files:", os.listdir('.'))

try:
    import mediapipe
    print("MediaPipe File Location:", mediapipe.__file__)
except Exception as e:
    print("Error importing:", str(e))
