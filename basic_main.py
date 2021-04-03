print('Start dl file...')

import sys
import matplotlib.pyplot as plt
from load_cifar_10_alt import load_data
#from keras.datasets import cifar10
# from load_cifar_10 import load_cifar_10_data

if sys.platform == "win32":
    cifar_dir = r'C:\Users\User\Documents\virtual\basic_dl\cifar-10-batches-py'
elif sys.platform == "linux":
    cifar_dir = r'/scratch/jamesang/proj_files/first_test/cifar-10-batches-py/'
else:
    pass

(x_train, y_train), (x_test, y_test) = load_data(cifar_dir)
print(x_train.shape)
print(y_train.shape)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train.shape
# y_train.shape

# cifar_train_data, _, cifar_train_labels, \
#     cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names = load_cifar_10_data(r'C:\Users\User\Documents\virtual\basic_dl\cifar-10-batches-py')
#
# print(cifar_train_data.shape)
# print(cifar_train_labels.shape)
# cifar_test_filenames

from keras.utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
plt.imshow(x_train[45])

# RESHAPING THE MATRIX SO THAT EACH IMAGE IS CONVERTED INTO SINGLE ROWS
x_traindata = x_train.reshape(50000,-1)
x_testdata = x_test.reshape(10000,-1)

# PREPROCSSING
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_traindatascaled = scaler.fit_transform(x_traindata)


# DEVELOPING THE MODEL ARCHITECTURE
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(150,
                activation='relu',
                input_dim = 3072))

model.add(Dense(150,
                activation='relu'
                ))

model.add(Dense(200,
                activation='relu'
                ))
model.add(Dense(10,
                activation = 'softmax'))

# COMPILING
from keras.optimizers import RMSprop
opt = RMSprop(learning_rate=0.00001)
model.compile(optimizer=opt,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
              )
import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')

# TRAINING
with tf.device('/gpu:0'):
    model.fit(x=x_traindata,
              y=y_train_cat,
              batch_size=64*2,
              shuffle=True,
              epochs=500)
scores =model.evaluate(x=x_testdata,
               y=y_test_cat)
