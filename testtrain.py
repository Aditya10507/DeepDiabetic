import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import ResNet152V2
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
labels = []
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def build_dataset_arrays():
    features = []
    targets = []
    for root, dirs, directory in os.walk(path):
        for filename in directory:
            name = os.path.basename(root)
            if 'Thumbs.db' in filename:
                continue
            img = cv2.imread(root + "/" + filename)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            features.append(img)
            targets.append(getLabel(name))
            print(name + " " + str(getLabel(name)))
    features = np.asarray(features)
    targets = np.asarray(targets)
    np.save(X_PATH, features)
    np.save(Y_PATH, targets)
    return features, targets

if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
    X = np.load(X_PATH)
    Y = np.load(Y_PATH)
else:
    X, Y = build_dataset_arrays()


X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
if os.path.exists(DATA_PATH):
    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, X_test, y_train, y_test = data
else:
    data = np.asarray([X_train, X_test, y_train, y_test], dtype=object)
    np.save(DATA_PATH, data)
print(X_test.shape)

efficient_base = EfficientNetB0(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in efficient_base.layers:
    layer.trainable = False

headModel = efficient_base.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel) # Consistent with VGG and ResNet
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel) # Consistent with VGG and ResNet
headModel = Dropout(0.3)(headModel) # Consistent with VGG and ResNet
headModel = Dense(units = y_train.shape[1], activation = 'softmax')
efficient_model = Model(inputs=efficient_base.input, outputs=headModel)
efficient_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
data_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

if os.path.exists(EFFICIENT_WEIGHTS_PATH) == False:
    model_check_point = ModelCheckpoint(filepath=EFFICIENT_WEIGHTS_PATH, verbose = 1, save_best_only = True)
    hist = efficient_model.fit(data_generator.flow(X_train, y_train, batch_size=32),
                               epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open(EFFICIENT_HISTORY_PATH, 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    efficient_model.load_weights(EFFICIENT_WEIGHTS_PATH)
predict = efficient_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)


vgg = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in vgg.layers:
    layer.trainable = False
headModel = vgg.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
vgg_model = Model(inputs=vgg.input, outputs=headModel)
vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists(VGG_WEIGHTS_PATH) == False:
    model_check_point = ModelCheckpoint(filepath=VGG_WEIGHTS_PATH, verbose = 1, save_best_only = True)
    hist = vgg_model.fit(data_generator.flow(X_train, y_train, batch_size=32),
                               epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open(VGG_HISTORY_PATH, 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vgg_model.load_weights(VGG_WEIGHTS_PATH)
predict = vgg_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test1, predict)
print(acc)

resnet = ResNet152V2(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
for layer in resnet.layers:
    resnet.trainable = False
headModel = resnet.output
headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
resnet_model = Model(inputs=resnet.input, outputs=headModel)
resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists(RESNET_WEIGHTS_PATH) == False:
    model_check_point = ModelCheckpoint(filepath=RESNET_WEIGHTS_PATH, verbose = 1, save_best_only = True)
    hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open(RESNET_HISTORY_PATH, 'wb')
    pickle.dump(hist.history, f)
    f.close()
else:
    resnet_model.load_weights(RESNET_WEIGHTS_PATH)    
    
predict = resnet_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test1, predict)
p = precision_score(y_test1, predict,average='macro') * 100
r = recall_score(y_test1, predict,average='macro') * 100
f = f1_score(y_test1, predict,average='macro') * 100
conf_matrix = confusion_matrix(y_test1, predict)
print(acc)
metric = np.asarray([acc, p, r, f])
np.save(METRIC_PATH, metric)
np.save(CM_PATH, conf_matrix)

metric = np.load(METRIC_PATH, allow_pickle=True)
print(metric)


conf_matrix = np.load(CM_PATH)
print(conf_matrix)



