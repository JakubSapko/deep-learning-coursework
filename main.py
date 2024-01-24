import pandas as pd
import numpy as np
import os

from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder

TRAIN_DIR = "./train"
TEST_DIR = "./test"


def load_dataset(directory):
    image_paths = []
    labels = []

    for label in os.listdir(directory):
        for filename in os.listdir(directory + "/" + label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)

        print(label, "Completed")

    return image_paths, labels


# convert into dataframe
train = pd.DataFrame()
train["image"], train["label"] = load_dataset(TRAIN_DIR)
# shuffle the dataset
train = train.sample(frac=1).reset_index(drop=True)
train.head()

test = pd.DataFrame()
test["image"], test["label"] = load_dataset(TEST_DIR)
test.head()


def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


train_features = extract_features(train["image"])
test_features = extract_features(test["image"])

# normalize the image
x_train = train_features / 255.0
x_test = test_features / 255.0
le = LabelEncoder()
le.fit(train["label"])
y_train = le.transform(train["label"])
y_test = le.transform(test["label"])
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)
y_train[0]
# config
input_shape = (48, 48, 1)
output_class = 7
model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
# output layer
model.add(Dense(output_class, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
history = model.fit(
    x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test)
)

model.save("model.keras")
