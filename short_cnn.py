import splitfolders
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras import metrics
from keras import callbacks
import time
import os
import pickle


def train_cnn_model(width, height, num_samples, needs_split=False):
    """
    This trains a cnn model and outputs the model to the specified location below
    :param width: width of image
    :param height: height of image
    :param num_samples: total number of samples that are being split
    :param needs_split: if the samples are split into the output folder or not
    """
    start = time.time()
    if needs_split:
        splitfolders.ratio("assets/cnn_images_5m/", output="assets/cnn_images_5m/output",
                           seed=0, ratio=(0.8, 0.1, 0.1), group_prefix=None)

    model_metrics = ['accuracy', metrics.BinaryCrossentropy(), metrics.Precision(),
                     metrics.Recall(), metrics.BinaryAccuracy()]
    img_width, img_height = width, height
    train_data_dir = 'assets/cnn_images_5m/output/train'
    val_data_dir = 'assets/cnn_images_5m/output/val'
    test_data_dir = 'assets/cnn_images_5m/output/test'
    epochs = 75
    batch_size = 32
    validation_steps = num_samples * 0.1 // batch_size
    steps_per_epoch = num_samples * 0.8 // batch_size,
    classes_num = 2
    nb_filters1 = 16
    nb_filters2 = 32
    conv1_size = 4
    conv2_size = 2
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
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=model_metrics)

    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    # buy is 0, hold is 1

    validation_generator = test_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    log_dir = './tf-log/'
    tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    cbks = [tb_cb]

    history = model.fit(
        train_generator,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks,
        validation_steps=validation_steps)

    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    model.save(f'assets/models/joey_cnn_intraday/cnn_model_5m_{epochs}epochs_{classes_num}classes.h5')
    model.save_weights(f'assets/models/joey_cnn_intraday/cnn_weights_5m_{epochs}epochs_{classes_num}classes.h5')

    with open(f'assets/models/joey_cnn_intraday/history_5m_{epochs}epochs_{classes_num}classes.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print('binary_accuracy', history.history['val_binary_accuracy'])
    print('precision', history.history['val_precision'])
    print('recall', history.history['val_recall'])

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


if __name__ == '__main__':
    needs_split = True
    if os.path.exists('assets/cnn_images_5m/output'):
        needs_split = False
    train_cnn_model(width=203, height=202, num_samples=64947, needs_split=needs_split)
