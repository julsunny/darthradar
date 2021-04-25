#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score as accuracy  # Our evaluation metric for this example.

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

tx_range = np.arange(3,11,2)
ty_range = np.arange(2,5)
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