
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
physical_devices

tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


import tensorflow as tf
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


import os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
