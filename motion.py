print("Script started")

try:
    import clr
    import sys
    import time
    import os
    import math
    print("Standard imports OK")

    KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"
    sys.path.append(KINESIS_PATH)

    clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
    print("DLL 1 OK")
    clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.GenericMotorCLI.dll"))
    print("DLL 2 OK")
    clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll"))
    print("DLL 3 OK")

    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
    print("Import 1 OK")
    from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import LongTravelStage
    print("Import 2 OK")
    from System import Decimal
    print("Import 3 OK")

except Exception as e:
    print("ERROR:", e)

input("Press Enter to exit...")"))

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import LongTravelStage
from System import Decimal

# ── Settings ──────────────────────────────────────────────
SERIAL_NUM   = "45540954"
CENTER       = 75.0    # Center position in mm
AMPLITUDE    = 30.0    # How far each side from center in mm (75±30 = 45mm to 105mm)
OSCILLATIONS = 5       # Number of full sine wave cycles
STEPS        = 100     # Steps per cycle (more = smoother)
PERIOD       = 4.0     # Time in seconds for one full cycle
# ──────────────────────────────────────────────────────────

def connect_device():
    DeviceManagerCLI.BuildDeviceList()
    print("Connected devices:", list(DeviceManagerCLI.GetDeviceList()))

    device = LongTravelStage.CreateLongTravelStage(SERIAL_NUM)
    device.Connect(SERIAL_NUM)

    if not device.IsSettingsInitialized():
        device.WaitForSettingsInitialized(5000)

    device.StartPolling(250)
    time.sleep(0.5)

    device.EnableDevice()
    time.sleep(0.5)

    device.LoadMotorConfiguration(SERIAL_NUM)

    return device

def home_device(device):
    print("Homing device...")
    device.Home(60000)
    print("Homing complete.\n")

def sinusoidal_oscillate(device):
    step_delay = PERIOD / STEPS

    print(f"Starting sinusoidal oscillation:")
    print(f"  Center: {CENTER} mm | Amplitude: ±{AMPLITUDE} mm")
    print(f"  Range: {CENTER - AMPLITUDE} mm to {CENTER + AMPLITUDE} mm")
    print(f"  Cycles: {OSCILLATIONS} | Period: {PERIOD}s | Steps per cycle: {STEPS}\n")

    for cycle in range(1, OSCILLATIONS + 1):
        print(f"Cycle {cycle}/{OSCILLATIONS}")