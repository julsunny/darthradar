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
from peakDetection import *

# Read the radar data into a variable.
data = h5py.File('data.h5', 'r')
X = []
y = []

for idx,element in enumerate(data['rdms']):
    X.append(data['rdms'][idx])
    number_targets = 0
    for target in data['labels'][str(idx)]:
        if target[4] != 3: number_targets += 1
    y.append(number_targets)

X_test = np.array(X)
y_test = np.array(y)

y_pred = np.zeros(y_test.shape)

tx_range = np.arange(10,11)
ty_range = np.arange(3,4)
gx_range = np.arange(2,3)
gy_range = np.arange(1,2)
rate_fa_range = np.logspace(-4, -0.5, 20)

maxscore = 0
argmax = []
for tx in tx_range:
    print("tx = ", tx)
    for ty in ty_range:
        for gx in gx_range:
            for gy in gy_range:
                for rate_fa in rate_fa_range:
                    i = 0
                    y_pred = np.zeros(y_test.shape)
                    for dmap in X_test:
                        dmap = cutout_middle_strip(dmap, 120, 136)
                        numpeaks, x, y, strength = detect_peaks(dmap, tx, ty, gx, gy, rate_fa)
                        y_pred[i] = numpeaks
                        i += 1
                    score = accuracy(y_test, y_pred)
                    if score > maxscore:
                        maxscore = score
                        argmax = [tx, ty, gx, gy, rate_fa]

print(maxscore)
print(argmax)

# Plot the confusion matrix.
#plot_confusion_matrix(model, X_test, y_test)
#plt.show()