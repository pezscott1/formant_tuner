import sounddevice as sd
print(sd.query_devices())
print("Default input:", sd.default.device)

import calibration.window
print(calibration.window.__file__)
