import clr
import sys
import os
import time

KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"
sys.path.append(KINESIS_PATH)

clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.GenericMotorCLI.dll"))
clr.AddReference(os.path.join(KINESIS_PATH, "Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll"))

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI

DeviceManagerCLI.BuildDeviceList()
time.sleep(3)
devices = DeviceManagerCLI.GetDeviceList()
print("Number of devices:", DeviceManagerCLI.GetDeviceListSize())
print("Connected devices:", list(devices))