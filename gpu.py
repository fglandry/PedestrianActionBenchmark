from tensorflow.python.client import device_lib
from tensorflow.test import gpu_device_name
print(device_lib.list_local_devices())
