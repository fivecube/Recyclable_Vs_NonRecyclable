from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import pickle
import imutils
import cv2
url = 'http://192.168.0.104:8080/photo.jpg'
# construct the argument parse and parse the arguments
MODEL_FILENAME = "smartbin_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "examples"

# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

garbage_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
# load the image
for image_file in garbage_image_files:
    print(image_file)
    image = cv2.imread(image_file)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (64, 64))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    prediction = model.predict(image)
    label = lb.inverse_transform(prediction)[0]
    # build the label
    # label = "recycle" if recycle > not_recycle else "Not recycle"
    # proba = recycle if recycle > not_recycle else not_recycle
    # label = "{}: {:.2f}%".format(label, proba * 100)
    label = "{}".format(label)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)
