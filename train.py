
import tensorflow.compat.v1 as tf

tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 

import os
print(os.getcwd())
#os.chdir("Project")
#print(os.getcwd())
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Read in label file and return a dictionary {'filename' : label}.
#
def import_labels(label_file):
    labels = dict()

    import csv
    with open(label_file) as fd:
        csvreader = csv.DictReader(fd)

        for row in csvreader:
            labels[row['filename']] = int(row['label'])
    return labels

import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image

import numpy as np
#import tensorflow as tf 

#tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 
#tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_root_dir, labels_dict, batch_size, target_dim, preprocess_func=None, use_augmentation=False):
        self._labels_dict = labels_dict
        self._img_root_dir = img_root_dir
        self._batch_size = batch_size
        self._target_dim = target_dim
        self._preprocess_func = preprocess_func
        self._n_classes = len(set(self._labels_dict.values()))
        self._fnames_all = list(self._labels_dict.keys())
        self._use_augmentation = use_augmentation

        if self._use_augmentation:
            self._augmentor = image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self._fnames_all)) / self._batch_size)

    def on_epoch_end(self):
        self._indices = np.arange(len(self._fnames_all))
        np.random.shuffle(self._indices)

    def __getitem__(self, index):
        indices = self._indices[index * self._batch_size:(index+1)*self._batch_size]

        fnames = [self._fnames_all[k] for k in indices]
        X,Y = self.__load_files__(fnames)

        return X,Y

    def __load_files__(self, batch_filenames):
        X = np.empty((self._batch_size, *self._target_dim, 3))
        Y = np.empty((self._batch_size), dtype=int)

        for idx, fname in enumerate(batch_filenames):
            img_path = os.path.join(self._img_root_dir, fname)
            img = image.load_img(img_path, target_size=self._target_dim)
            x = image.img_to_array(img)
            if self._preprocess_func is not None:
                x = self._preprocess_func(x)

            X[idx,:] = x 
            Y[idx] = self._labels_dict[fname]-1

        if self._use_augmentation:
            it = self._augmentor.flow(X, batch_size=self._batch_size, shuffle=False)
            X = it.next()

        if self._preprocess_func is not None:
            X = self._preprocess_func(X)

        return X, tf.keras.utils.to_categorical(Y, num_classes=self._n_classes)

from tensorflow.keras.utils import to_categorical
y_train = import_labels("train_labels.csv")
y_test = import_labels("test_labels.csv")
batch_size=32
image_size = (496, 496)

def preprocess_func(x):
    return x / 255.0

datagen_train = DataGenerator('train', y_train, batch_size, image_size, preprocess_func=preprocess_func, use_augmentation=True)
datagen_test = DataGenerator('test', y_test, batch_size, image_size, preprocess_func=preprocess_func, use_augmentation=True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# Design model
model = keras.Sequential()
model.add(layers.Input(image_size + (3,)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPool2D((3,3)))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPool2D((5,5)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPool2D((4,4)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
model.add(layers.MaxPool2D((3,3)))
model.add(layers.BatchNormalization())
#model.add(layers.Dropout(0.5))
#model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation="relu"))
#model.add(layers.BatchNormalization())
#model.add(layers.Dense(16, activation="relu"))
#model.add(layers.BatchNormalization())
model.add(layers.Dense(102, activation="softmax"))
model.compile(optimizer=Adam(learning_rate=0.02), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Train model on dataset
model.fit(datagen_train, epochs=5,callbacks=[cp_callback
                                                                          # ,tensorboard_callback
                                                                          ])