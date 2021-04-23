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

# Read the radar data into a variable.
data = h5py.File('data.h5', 'r')

# Prepare an empty list for the features and labels.
Xy = []

# Go through every radar frame.
for idx, element in enumerate(data['rdms']):
    # Prepare an empty list for storing the radar frame.
    # This is required since the radar frame comes in as a matrix but for this example
    # it should be saved as vector.
    rdms_row = []
    # Prepare an empty list for the names of the columns in the Pandas DataFrame.
    column_names = []
    # The radar frame is a matrix. Go through every row of this matrix...
    for r in element:
        # ...and through every column.
        for c in r:
            # Save the value c at the current row and column to the radar frame vector.
            rdms_row.append(c)
            # Add a name for this column.
            column_names.append('rdms-' + str(len(rdms_row)))
    # Calculate the label, the number of targets.
    number_targets = 0
    for target in data['labels'][str(idx)]:
        # only count target if class is not 3 (no object)
        if target[4] != 3: number_targets += 1
    # Add the label to the current row (will be separated again later).
    rdms_row.append(number_targets)
    column_names.append('number_targets')
    # Add the prepared row to our feature/label list.
    Xy.append(rdms_row)

# Create a Pandas DataFrame from our feature/label list Xy.
df = pd.DataFrame(Xy, columns=column_names)

# Define our features X as all columns of Xy excluding the label column.
X = df.drop(['number_targets'], axis='columns')
# Define the label column of Xy as our label.
y = df.number_targets

# Normalize X
X = preprocessing.normalize(X, norm='l2')

# Split X and y into training and testing data each.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Prepare the classifier.
model = MLPClassifier(solver='lbfgs', max_iter=1000)

# Train the classifier (might take some minutes).
print("Start training...")
model.fit(X_train, y_train)

# Predict values with the trained model.
y_pred = model.predict(X_test)

# Evaluate the prediction performance using the accuracy metric and print the result.
score = accuracy(y_test, y_pred)
print(score)

# Plot the confusion matrix.
#plot_confusion_matrix(model, X_test, y_test)
#plt.show()