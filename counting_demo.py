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
i = 0

for dmap in X_test:
    dmap = cutout_middle_strip(dmap, 120, 136)
    numpeaks, x, y, strength = detect_peaks(dmap, tx=20, ty=3, gx=2, gy=1, rate_fa=0.15)
    y_pred[i] = numpeaks
    i += 1

score = accuracy(y_test, y_pred)
print("Counting Accuracy: ", score)