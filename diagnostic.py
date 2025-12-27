import sounddevice as sd
import calibration.window

print(sd.query_devices())
print("Default input:", sd.default.device)
print(calibration.window.__file__)
