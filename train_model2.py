import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras import backend
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import matplotlib.pyplot as plt
from imutils import paths
import random
import cv2
import os
import pickle

batch_size = 32
epochs = 50
iterations = 50
num_classes = 5
mean = [125.307, 122.95, 113.865]
std = [62.9932, 62.0887, 66.7048]
height, width, depth = (32, 48, 3)

GARBAGE_IMAGES_FOLDER = "dataset1"
MODEL_LABELS_FILENAME = "model_labels.dat"

input_shape = (height, width, depth)
# if we are using "channels first", update the input shape
if backend.image_data_format() == "channels_first":
    input_shape = (depth, height, width)


def get_data():
    data = []
    labels = []
    print('Reading images from dataset...')

    image_paths = list(paths.list_images(GARBAGE_IMAGES_FOLDER))
    random.seed(42)
    random.shuffle(image_paths)
    # loop over the input images
    for image_file in image_paths:
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]
        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype=np.float32)
    return data, labels


def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation='relu',
                     kernel_initializer='he_normal', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001


if __name__ == '__main__':

    # load data
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    data, labels = get_data()
    (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.3, random_state=42)
    lb = LabelBinarizer().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # data preprocessing  [raw - mean / std]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    # build network
    model = build_model()
    print(model.summary())
    # set callback
    tb_cb = TensorBoard(log_dir='./lenet_dp_da', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start train
    H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=iterations,
                            epochs=epochs,
                            callbacks=cbks,
                            validation_data=(x_test, y_test))
    # save model
    model.save('lenet_dp_da.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Recycle/Not recycle")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('plot.png')
