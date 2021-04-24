#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This example reads the range Doppler image (256 x 32 matrix) and converts it to a vector with 8192
elements. This vector is used as feature vector X. The value y we want to learn in this simple
example is the number of targets within a certain radar frame. Because of the low amount of
potential outcomes this can be seen as classification problem. For demonstration the MLP Classifier
of the Sklearn library is used and a confusion matrix is plotted.
"""

"""
Dependencies: pandas, h5py, sklearn, matplotlib
install with: pip3 install pandas h5py sklearn matplotlib
"""

import h5py # Required to read the radar data.
import pandas as pd # Helps to represent the data in the correct format.
from sklearn.model_selection import train_test_split # Automatically split train and test data.
from sklearn import preprocessing # Preprocessing module to normalize X data
from sklearn.neural_network import MLPClassifier # The Multi-layer Perceptron classifier.
from sklearn.metrics import accuracy_score as accuracy # Our evaluation metric for this example.
from sklearn.metrics import plot_confusion_matrix # Module for plotting the confusion matrix.
import matplotlib.pyplot as plt # Default plotting module for Python.
import numpy as np
import scipy.ndimage

# Read the radar data into a variable.
data = h5py.File('data.h5', 'r')
X = data['rdms']
y = data['labels']
# Split X and y into training and testing data each.
# Normalize X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Predict values with the trained model.
y_pred_cfar = np.zeros(y_test.shape)

cutout_lower = 126
cutout_upper = 130
i = 0
for dmap in X_test.to_numpy():
    dmap = dmap.reshape(256,32)
    dmap = np.delete(dmap, np.arange(cutout_lower, cutout_upper + 1), axis=0)
    #plt.imshow(dmap.T, origin='lower', interpolation='bilinear', cmap='viridis')  # , aspect=256/32*0.5)
    #plt.draw()
    peaks = detect_peaks(dmap, tx=10, ty=3, gx=2, gy=1, rate_fa=1e-3)
    x, y = np.argwhere(peaks == 1).T
    plt.scatter(x, y, color='red')
    y_pred_cfar[i] = x.size
    i+=1
    #plt.pause(0.5)
    #plt.clf()

print(y_pred_cfar)
print(y_pred_cfar.shape)
#Prepare the classifier.
#model = MLPClassifier(solver='lbfgs', max_iter=1000)

# Train the classifier (might take some minutes).
#print("Start training...")
#model.fit(X_train, y_train)

# Evaluate the prediction performance using the accuracy metric and print the result.
#score1 = accuracy(y_test, y_pred_nn)
#print(score1)
score2 = accuracy(y_test, y_pred_cfar)
print(score2)

# Plot the confusion matrix.
#plot_confusion_matrix(model, X_test, y_test)
#plt.show()