import keras.preprocessing.image_dataset
import pandas as pd
import splitfolders
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras import callbacks
import time
import os

# HAS TO BE RUN ON COMMAND LINE UBUNTU 20.04 to use CUDA - venv install broken at the moment
# - Joey

results = pd.read_csv('assets/cnn_images_results.csv')
# splitfolders.ratio("assets/cnn_images/", output="assets/cnn_images/output",
#                    seed=0, ratio=(0.8, 0.1, 0.1), group_prefix=None)

img_width, img_height = 203, 202
train_data_dir = 'assets/cnn_images/output/train'
val_data_dir = 'assets/cnn_images/output/val'
test_data_dir = 'assets/cnn_images/output/test'
epochs = 100
validation_steps = 300
batch_size = 32
num_train_samples = 49970
num_val_samples = 6245
classes_num = 3  # increase, decrease, same
nb_filters1 = 32
nb_filters2 = 32
nb_filters3 = 64
conv1_size = 3
conv2_size = 2
conv3_size = 5
pool_size = 2

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()
    # rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True)

test_datagen = ImageDataGenerator() # rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit(
    train_generator,
    batch_size=batch_size,
    steps_per_epoch=256,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

target_dir = './models/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save('assets/models/cnn_model.h5')
model.save_weights('assets/models/cnn_weights.h5')

# Calculate execution time
end = time.time()
dur = end - start

if dur < 60:
    print("Execution Time:", dur, "seconds")
elif 60 < dur < 3600:
    dur = dur / 60
    print("Execution Time:", dur, "minutes")
else:
    dur = dur / (60 * 60)
    print("Execution Time:", dur, "hours")
