import cv2
import pickle
import os
import random
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import img_to_array
from keras import backend
import matplotlib.pyplot as plt

height, width, depth = (32, 42, 3)
classes = 5
input_shape = (height, width, depth)
# if we are using "channels first", update the input shape
if backend.image_data_format() == "channels_first":
    input_shape = (depth, height, width)

GARBAGE_IMAGES_FOLDER = "dataset1"
MODEL_FILENAME = "smartbin_model3.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
BS = 32
EPOCHS = 25


# initialize the data and labels
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

print('Number of images:', len(data))
print('Labels:', set(labels))

# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print('Splitting dataset into Train and Test...')
# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.3, random_state=42)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings.
# We'll need this later when we use the model to decode what it's predictions mean
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(classes, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(aug.flow(X_train, Y_train, batch_size=BS),
                        validation_data=(X_test, Y_test), steps_per_epoch=len(X_train) // BS,
                        epochs=EPOCHS, verbose=1)
# Train the neural network
# H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=BS, epochs=EPOCHS, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Recycle/Not recycle")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')
