#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from dataloader import RadarImageTargetSet
from sklearn.model_selection import StratifiedKFold

###
### INPUT PREPARATION
###

# load dataset
ds = RadarImageTargetSet()

# we use the box labels as given in the dataset
# and train a simple neural network to classify
# them

boxes = []
for (img, tgt) in ds:
    for box in tgt['boxes']:
        boxes.append(box)
boxes = np.array(boxes)
box_xsizes = boxes[:, 2] - boxes[:, 0]
box_ysizes = boxes[:, 3] - boxes[:, 1]
max_x_size = int(max(box_xsizes))
max_y_size = int(max(box_ysizes))

# the maximum box widths / heights are used as the
# input layer dimensions to the convolutional layer
# this ensures the biggest box will fit, the smaller
# boxes will be padded with zeros
INPUT_X_SIZE = max_x_size
INPUT_Y_SIZE = max_y_size

print(INPUT_X_SIZE, INPUT_Y_SIZE)

# we cut each labeled box out of the image
# and pad it to the input size

# box cutouts
x = []
# box center positions in radar frame
c = []
# classes in one hot encoding
y = []

for (img, tgt) in ds:
    #print(img.shape)
    for (class_type, (y0, x0, y1, x1)) in zip(tgt['labels'], list(tgt['boxes'])):
        if np.isclose(class_type, 3.0):
            # don't include "no objects"
            continue
        c.append([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        # cut box out of image
        #print(class_type)
        #print(x0, y0, x1, y1)
        cutout = img[int(x0):int(x1), int(y0):int(y1)]
        #print(cutout.shape)
        if cutout.shape[0] > INPUT_X_SIZE and cutout.shape[1] > INPUT_Y_SIZE:
            print("can't process box of shape", x0, y0, x1, y1, cutout.shape)
            continue
        # padding putting it in upper left corner
        # TODO: center
        y_padding = int(INPUT_Y_SIZE - cutout.shape[0])
        x_padding = int(INPUT_X_SIZE - cutout.shape[1])
        padded = np.pad(cutout, ((y_padding // 2, y_padding - (y_padding // 2)), (x_padding // 2, x_padding - (x_padding // 2))), mode='constant', constant_values=0.0)
        x.append(padded)
        # class as one hot encoding
        y.append([1.0 if int(class_type) == i else 0.0 for i in range(3)])

x = np.array(x)
c = np.array(c)
y = np.array(y)
print("box cutouts shape:", x.shape)
print("box centers shape:", c.shape)
print("classes in one hot encoding shape:", y.shape)

# we split the training data into a training and test set

from sklearn.model_selection import train_test_split 
x_train,x_test,c_train,c_test,y_train,y_test=train_test_split(x, c, y, test_size=0.33, random_state=42)

x_train=np.asarray(x_train)
c_train=np.asarray(c_train)
y_train=np.asarray(y_train)
x_test=np.asarray(x_test)
c_test=np.asarray(c_test)
y_test=np.asarray(y_test)
x_train=np.reshape(x_train,[-1,INPUT_X_SIZE,INPUT_Y_SIZE,1])
c_train=np.reshape(c_train,[-1,2,1])
x_test=np.reshape(x_test,[-1,INPUT_X_SIZE,INPUT_Y_SIZE,1])

###
### THE MODEL
###

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import Dense, Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

##################################
#
#   We create the following
#   network archtiecture:
#
#    INPUT1             INPUT2
#  "box center"      "box cutout"
# (float, float)    2d np.ndarray
#      |                  |
#      |                 CNN
#      |                  |
#      |               Flatten
#      \______concat______/
#               |
#             Dense
#               |
#           3 Classes
#       (multiclass output)
#################################

# define two sets of inputs
box_cutout_input = Input(shape=(INPUT_X_SIZE,INPUT_Y_SIZE,1))
box_center_input = Input(shape=(2,))

# the right branch operates on the first input
# (the image data)
x = k.layers.Conv2D(32,3,3,padding='same',
    dilation_rate=(1, 1),
    activation="relu")(box_cutout_input)
x = k.layers.Conv2D(32,3,3,padding='same',
    dilation_rate=(1, 1),
    activation="relu")(x)
x = k.layers.Flatten()(x)
x = Model(inputs=box_cutout_input, outputs=x)

# the left branch just passes through the
# center coordinates of the box
y = Dense(1, activation="linear")(box_center_input)
y = Model(inputs=box_center_input, outputs=y)

# after flatting the 2d output from the convolutional subnetwork,
# we add the box center coordinates to the inputs to the fully
# connected part
combined = concatenate([x.output, y.output])

# a few fully connected layers
z = Dense(64, activation="relu")(combined)
z = Dense(64, activation="relu")(z)
z = Dense(64, activation="relu")(z)

# the final output which gives confidence with respect
# to the three classes
z = Dense(3, activation="softmax")(z)

# compilation and print summary
model = Model(inputs=[x.input, y.input], outputs=z)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

###
### TRAINING
### 

EPOCHS = 100
history=model.fit(x = [x_train, c_train], y = y_train, epochs=EPOCHS,batch_size=374, validation_data=([x_test, c_test], y_test)) 
# set batch size to number of images in dataset ==> slower training but minority class is consicerd in every parameter update

# plot accuracy over training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plot loss over training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.ylim((0, 10))
plt.show()

###
### VALIDATION
###

from sklearn.metrics import confusion_matrix
# predict classes on the test set
y_pred=history.model.predict([x_test, c_test])

# argmax converts from multiclass output to
# single class, by giving the class to which
# the model assigned the highest confidence
y_pred_abs = np.argmax(y_pred, axis=1)
y_test_abs = np.argmax(y_test, axis=1)
# create confusion matrix
cm=confusion_matrix(y_test_abs,y_pred_abs)
plt.imshow(cm, cmap = 'jet')
plt.title("confusion matrix")
plt.show()
print("confusion matrix:")
print(cm)