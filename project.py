# importing libraries
import os
import random
import matplotlib.pyplot as plt
import keras
from keras.callbacks import Callback
from imblearn.over_sampling import SMOTE
import keras
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.datasets import make_classification
from imblearn.over_sampling import KMeansSMOTE
 
# class for accuracy and loss functions
class Accuracy_Loss(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.accuracy_values = []
        self.loss_values = []
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["accuracy"]
        loss = logs["loss"]
        self.accuracy_values.append(accuracy)
        self.loss_values.append(loss)

# preprocessing function
def image_preprocess (img,img_size):
    img_resize = cv2.resize(img, (img_size, img_size))
    img_edge = cv2.Canny(img_resize, 100, 300) + img_resize
    img_blur = cv2.GaussianBlur(img_edge, (3, 3), 0)
    return img_blur

img_size = 256
# change this directory to where you have saved our code
directory = r"D:\users\Talin\Documents\image processing\computer_vision\project\ct-scans\The IQ-OTHNCCD lung cancer dataset"
categories = ["Benign cases", "Malignant cases", "Normal cases"]

# displaying one sample from each class
for i in categories:
    path = os.path.join(directory, i)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        img = cv2.imread(filepath, 0)
        cv2.imshow(str(i), img)
        img_preprocess = image_preprocess(img,img_size)
        cv2.imshow("image preprocessed",img_preprocess)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break


# functoin for reading data and giving them labels
def read_data(categories, directory, img_size):
    data = []
    for i in categories:
        path = os.path.join(directory, i)
        class_num = categories.index(i)
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            img = cv2.imread(filepath, 0)
            img = image_preprocess(img, img_size)
            data.append([img, class_num])
    return data

# reading data
data = read_data(categories, directory, img_size)
random.shuffle(data)
X, y = [], []
for feature, label in data:
    X.append(feature)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 1)/255
y = np.array(y)
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4, stratify=y)

#-------------- Augmentation:
# Define the data generators
train_datagen = ImageDataGenerator(
                              rotation_range=10,
                              shear_range=0.1,
                            horizontal_flip=True,
                             fill_mode='nearest')
train_generator = train_datagen.flow(X_train, y_train,batch_size=32)


#------------- SMOTE
''' in order to use method replace any X_train
    and y_train with X_train_SMOTE and 
    y_train_SMOTE respectively 
'''
# in order for SMOTE ro work X_train must be 2D
X_train_reshaped = X_train.reshape(X_train.shape[0], img_size*img_size*1)
#print(Counter(y_train))
smote = SMOTE()
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_reshaped, y_train)
#print(Counter(y_train_SMOTE))
# transform X_train back to 4D for the training process
X_train_SMOTE = X_train_SMOTE.reshape(X_train_SMOTE.shape[0], img_size, img_size, 1)

#------------- kMeanSMOTE
''' in order to use method replace any X_train
    and y_train with X_train_KMeanSMOTE and 
    y_train_KMeanSMOTE respectively 
'''
# in order for KMeanSMOTE ro work X_train must be 2D
#X_train_reshaped = X_train.reshape(X_train.shape[0], img_size*img_size*1)
#print(Counter(y_train))
# if the ratio in a cluster is 10% benign images Smote is used
kMeansmote = KMeansSMOTE(cluster_balance_threshold=0.1, random_state=42)
X_train_KMeanSMOTE, y_train_KMeanSMOTE = kMeansmote.fit_resample(X_train_reshaped, y_train)
#print(Counter(y_train_KMeanSMOTE))
# transform X_train back to 4D for the training process
X_train_KMeanSMOTE = X_train_KMeanSMOTE.reshape(X_train_KMeanSMOTE.shape[0], img_size, img_size, 1)

#------------- ADASYN
''' in order to use method replace any X_train
    and y_train with X_train_ADASYN and 
    y_train_ADASYN respectively 
'''
#print (Counter(y_train))
adasyn = ADASYN(random_state=42)
#Reshape the original samples to remove the batch dimension
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
# Apply ADASYN to generate synthetic samples
X_train_ADASYN, y_train_ADASYN = adasyn.fit_resample(X_train_reshaped, y_train)
#print (Counter(y_train_sampled))
# transform X_train back to 4D for the training process
X_train_ADASYN = X_train_ADASYN.reshape(X_train_ADASYN.shape[0], img_size, img_size, 1)

#------------- weights
''' in order to use this method just this line
    , class_weight=weights
    to the end of the model.fit line
'''
weights = {
    0: X_train.shape[0]/(2*Counter(y_train)[0]),
    1: X_train.shape[0]/(2*Counter(y_train)[1]),
    2: X_train.shape[0]/(2*Counter(y_train)[2]),
}

# building the model
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
# Dropout rate of 0.2 during training
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
accuracy_loss = Accuracy_Loss()
fit = model.fit(X_train, y_train, batch_size=5, epochs=10, validation_data=(X_test, y_test), callbacks=[accuracy_loss])

''' in order to use data augmentation
    you need to uncomment this part
    and also comment the fit line above
'''
#fit = model.fit(train_generator,
#          steps_per_epoch=len(X_train) / 32,
#          epochs=50,
#          validation_data=(X_test, y_test),
#          validation_steps=len(X_test) / 32,
#          callbacks=[accuracy_loss])

# training accuracy/loss vs number of epochs
accuracy = accuracy_loss.accuracy_values
loss = accuracy_loss.loss_values
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, label="Training accuracy")
plt.plot(epochs, loss, label="Training loss")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Accuracy and Loss")
plt.legend()
plt.show()

# predicting on photos not in the database
# change this directory to where you have saved our code
direc = r"D:\users\Talin\Documents\image processing\computer_vision\project\ct-scans\data for predicting"
pred = read_data(categories, direc, img_size)
random.shuffle(pred)
pred_img, pred_label = [], []
for feature, label in pred:
    pred_img.append(feature)
    pred_label.append(label)
pred_img = np.array(pred_img).reshape(-1, img_size, img_size, 1)/255
actual_labels = np.array(pred_label)
predictions = model.predict(pred_img)
class_labels = ["Benign cases", "Malignant cases", "Normal cases"]
predicted_labels = np.argmax(predictions, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(actual_labels, predicted_labels)
# plotting the matrix as a graph
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)
# Customize confusion matrix plot
ax.set(xticks=np.arange(conf_matrix.shape[1]),
       yticks=np.arange(conf_matrix.shape[0]),
       xticklabels=class_labels,
       yticklabels=class_labels,
       xlabel='Predicted',
       ylabel='Actual',
       title='Confusion Matrix')
# Add text annotations for each cell in the confusion matrix
thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
plt.show()