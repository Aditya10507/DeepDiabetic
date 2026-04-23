import os
import pickle

import cv2
import numpy as np


path = "Dataset"
IMAGE_SIZE = (224, 224)
X_PATH = "model/X_224.npy"
Y_PATH = "model/Y_224.npy"
DATA_PATH = "model/data_224.npy"
EFFICIENT_WEIGHTS_PATH = "model/efficient_weights_224.hdf5"
VGG_WEIGHTS_PATH = "model/vgg_weights_224.hdf5"
RESNET_WEIGHTS_PATH = "model/resnet_weights_224.hdf5"
EFFICIENT_HISTORY_PATH = "model/efficient_history_224.pckl"
VGG_HISTORY_PATH = "model/vgg_history_224.pckl"
RESNET_HISTORY_PATH = "model/resnet_history_224.pckl"
METRIC_PATH = "model/metric_224.npy"
CM_PATH = "model/cm_224.npy"


def discover_labels(dataset_path):
    labels = []
    for root, _dirs, files in os.walk(dataset_path):
        for _ in files:
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    return labels


def get_label(name, labels):
    for index, label in enumerate(labels):
        if label == name:
            return index
    return -1


def build_dataset_arrays(dataset_path, labels):
    features = []
    targets = []
    for root, _dirs, files in os.walk(dataset_path):
        for filename in files:
            if "Thumbs.db" in filename:
                continue
            name = os.path.basename(root)
            image = cv2.imread(os.path.join(root, filename))
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            features.append(image)
            targets.append(get_label(name, labels))
            print(name + " " + str(get_label(name, labels)))
    features = np.asarray(features)
    targets = np.asarray(targets)
    np.save(X_PATH, features)
    np.save(Y_PATH, targets)
    return features, targets


def main():
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications import ResNet152V2
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import AveragePooling2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import to_categorical

    labels = discover_labels(path)

    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        X = np.load(X_PATH)
        Y = np.load(Y_PATH)
    else:
        X, Y = build_dataset_arrays(path, labels)

    X = X.astype("float32")
    X = X / 255

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X_train, X_test, y_train, y_test = data
    else:
        data = np.asarray([X_train, X_test, y_train, y_test], dtype=object)
        np.save(DATA_PATH, data)
    print(X_test.shape)

    efficient_base = EfficientNetB0(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
        include_top=False,
        weights="imagenet",
    )
    for layer in efficient_base.layers:
        layer.trainable = False

    head_model = efficient_base.output
    head_model = AveragePooling2D(pool_size=(1, 1))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.3)(head_model)
    head_model = Dense(units=y_train.shape[1], activation="softmax")(head_model)
    efficient_model = Model(inputs=efficient_base.input, outputs=head_model)
    efficient_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    data_generator = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    if not os.path.exists(EFFICIENT_WEIGHTS_PATH):
        model_check_point = ModelCheckpoint(filepath=EFFICIENT_WEIGHTS_PATH, verbose=1, save_best_only=True)
        hist = efficient_model.fit(
            data_generator.flow(X_train, y_train, batch_size=32),
            epochs=40,
            validation_data=(X_test, y_test),
            callbacks=[model_check_point],
            verbose=1,
        )
        with open(EFFICIENT_HISTORY_PATH, "wb") as file:
            pickle.dump(hist.history, file)
    else:
        efficient_model.load_weights(EFFICIENT_WEIGHTS_PATH)
    predict = efficient_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test1, predict)
    print(acc)

    vgg = VGG16(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
        include_top=False,
        weights="imagenet",
    )
    for layer in vgg.layers:
        layer.trainable = False
    head_model = vgg.output
    head_model = AveragePooling2D(pool_size=(1, 1))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.3)(head_model)
    head_model = Dense(y_train.shape[1], activation="softmax")(head_model)
    vgg_model = Model(inputs=vgg.input, outputs=head_model)
    vgg_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if not os.path.exists(VGG_WEIGHTS_PATH):
        model_check_point = ModelCheckpoint(filepath=VGG_WEIGHTS_PATH, verbose=1, save_best_only=True)
        hist = vgg_model.fit(
            data_generator.flow(X_train, y_train, batch_size=32),
            epochs=40,
            validation_data=(X_test, y_test),
            callbacks=[model_check_point],
            verbose=1,
        )
        with open(VGG_HISTORY_PATH, "wb") as file:
            pickle.dump(hist.history, file)
    else:
        vgg_model.load_weights(VGG_WEIGHTS_PATH)
    predict = vgg_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test1, predict)
    print(acc)

    resnet = ResNet152V2(
        input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
        include_top=False,
        weights="imagenet",
    )
    for layer in resnet.layers:
        layer.trainable = False
    head_model = resnet.output
    head_model = AveragePooling2D(pool_size=(1, 1))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.3)(head_model)
    head_model = Dense(y_train.shape[1], activation="softmax")(head_model)
    resnet_model = Model(inputs=resnet.input, outputs=head_model)
    resnet_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    if not os.path.exists(RESNET_WEIGHTS_PATH):
        model_check_point = ModelCheckpoint(filepath=RESNET_WEIGHTS_PATH, verbose=1, save_best_only=True)
        hist = resnet_model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=40,
            validation_data=(X_test, y_test),
            callbacks=[model_check_point],
            verbose=1,
        )
        with open(RESNET_HISTORY_PATH, "wb") as file:
            pickle.dump(hist.history, file)
    else:
        resnet_model.load_weights(RESNET_WEIGHTS_PATH)

    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_test1, predict)
    precision = precision_score(y_test1, predict, average="macro") * 100
    recall = recall_score(y_test1, predict, average="macro") * 100
    fscore = f1_score(y_test1, predict, average="macro") * 100
    conf_matrix = confusion_matrix(y_test1, predict)
    print(acc)
    metric = np.asarray([acc, precision, recall, fscore])
    np.save(METRIC_PATH, metric)
    np.save(CM_PATH, conf_matrix)

    metric = np.load(METRIC_PATH, allow_pickle=True)
    print(metric)

    conf_matrix = np.load(CM_PATH)
    print(conf_matrix)


if __name__ == "__main__":
    main()
