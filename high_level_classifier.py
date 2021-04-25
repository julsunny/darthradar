import h5py # Required to read the radar data.
import numpy as np
from sklearn.model_selection import train_test_split # Automatically split train and test data.
from sklearn import preprocessing # Preprocessing module to normalize X data
from sklearn.neural_network import MLPClassifier # The Multi-layer Perceptron classifier.
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score as accuracy # Our evaluation metric for this example.
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from peakDetection import *
from sklearn.metrics import plot_confusion_matrix # Module for plotting the confusion matrix.

STANDARD_BOXSIZE_X = 11
STANDARD_BOXSIZE_Y = 7
data = h5py.File('data.h5', 'r')
def generate_train_test_set(split):
    # Read the radar data into a variable.
    stats = []
    labels = []
    boxes = []
    imageNo = []

    for idx, element in enumerate(data['rdms']):
        for target in data['labels'][str(idx)]:
            if target[4] != 3:
                imageNo.append(idx)
                labels.append(target[4])
                x = np.array([int(0.5*(target[1]+target[3]))])
                y = np.array([int(0.5*(target[0]+target[2]))])
                d = cutout_middle_strip(element, 122, 134)
                b = return_static_box_bounds(x, y, d, STANDARD_BOXSIZE_X, STANDARD_BOXSIZE_Y)
                s = return_box_stats(d, b)[0]
                stats.append(s)
                boxes.append(b[0])

    stats = np.array(stats)
    labels = np.stack((np.array(imageNo), np.array(labels), np.array(boxes)[:,0], np.array(boxes)[:,1],
                       np.array(boxes)[:,2], np.array(boxes)[:,3]), axis = 1)
    stats_train, stats_test, labels_train, labels_test = train_test_split(stats, labels, test_size=split)

    return stats_train, stats_test, labels_train, labels_test

def train_model(stats_train, labels_train):
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(stats_train, labels_train)

    return model

def test_model_class_only(model, stats_test, labels_test, conf_matrix = True):
    labels_pred = model.predict(stats_test)
    score = accuracy(labels_pred, labels_test[:,1])
    if conf_matrix == True:
        plot_confusion_matrix(model, stats_test, labels_test[:,1])
        plt.show()
    return score


def test_model_endtoend(model, stats_test, labels_test):
    img_data = {}
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for label in labels_test:
        img_data[str(label[0])] = []

    for label in labels_test:
        img_id = label[0]
        img_data[str(img_id)].append((label[1], label[2:5]))

    for key, val in img_data.items():
        d = cutout_middle_strip(data['rdms'][float(key)], 122, 134)
        numpeaks, x, y, strength = detect_peaks(d, tx=20, ty=3, gx=2, gy=1, rate_fa=0.15)
        bounds = return_static_box_bounds(x, y, d, STANDARD_BOXSIZE_X, STANDARD_BOXSIZE_Y)
        stats_pred = return_box_stats(d, bounds)

        overlap_matrix = np.zeros((len(val), len(bounds)))

        for l in range(len(val)):
            for b in range(len(bounds)):
                overlap_matrix[l,b] = bounds_overlap(val[l][1], bounds[b])

        pred_status = np.zeros(len(bounds))

        for l in range(len(val)):
            already_matched = False
            ones_encountered = False
            for b in range(len(bounds)):
                if (overlap_matrix[l, b] == 1):
                    ones_encountered = True
                    agree = (model.predict(stats_pred[b:b+1])[0] == val[l][0])
                    if agree == True:
                        if already_matched == True:
                            false_pos += 1
                        else:
                            true_pos += 1
                            already_matched = True
                    else:
                        false_pos += 1
            if ones_encountered == False:
                false_neg += 1

        for b in range(len(bounds)):
            not_matched = True
            for b in range(len(bounds)):
                if (overlap_matrix[l, b] == 1):
                    not_matched = False
            if not_matched == True:
                false_pos += 1

    return true_pos, false_pos, false_neg

def test_mean_performance(runs):
    score = 0
    for i in range(runs):
        stats_train, stats_test, labels_train, labels_test = generate_train_test_set(0.3)
        model = train_model(stats_train, labels_train[:, 1])
        score += test_model_class_only(model, stats_test, labels_test, conf_matrix = False)/runs

    print("Mean Accuracy (classifier only): ", score)

    true_pos_mean = 0
    false_pos_mean = 0
    false_neg_mean = 0
    accuracy_mean = 0
    for i in range(runs):
        stats_train, stats_test, labels_train, labels_test = generate_train_test_set(0.3)
        model = train_model(stats_train, labels_train[:, 1])
        true_pos, false_pos, false_neg = test_model_endtoend(model, stats_test, labels_test)
        true_pos_mean += true_pos/runs
        false_pos_mean += false_pos/runs
        false_neg_mean += false_neg/runs
        accuracy_mean += true_pos/((true_pos + false_pos + false_neg)*runs)

    print("Mean True Positives (end-to-end): ", true_pos_mean)
    print("Mean False Positives (end-to-end): ", false_pos_mean)
    print("Mean False Positives (end-to-end): ", false_neg_mean)
    print("-----------------------------------")
    print("Mean Accuracy (end-to-end): ", accuracy_mean)

def classification_demo(delay):
    # Read the radar data into a variable.

    classes = {
        '0': 'pedestrian',
        '1': 'car',
        '2': 'truck',
        '3': 'no object'
    }

    stats_train, stats_test, labels_train, labels_test = generate_train_test_set(0.3)
    model = train_model(stats_train, labels_train[:, 1])

    fig, ax = plt.subplots()

    for idx,dmap in enumerate(data['rdms']):
        ax.clear()
        dmap = cutout_middle_strip(dmap, 122, 134)
        numpeaks, x, y, strength = detect_peaks(dmap, tx=20, ty=3, gx=2, gy=1, rate_fa=0.15)
        bounds = return_static_box_bounds(x, y, dmap, STANDARD_BOXSIZE_X, STANDARD_BOXSIZE_Y)
        stats_pred = return_box_stats(dmap, bounds)
        greens = np.zeros(len(stats_pred), dtype=int)
        reds = np.zeros(len(stats_pred), dtype=int)
        blues = np.zeros(len(stats_pred), dtype=int)
        for i in range(len(stats_pred)):
            c = int(model.predict(np.array([stats_pred[i]]))[0])
            if c == 0:
                blues[i] = 1
            elif c == 1:
                reds[i] = 1
            else:
                greens[i] = 1

        plt.imshow(dmap.T, origin='lower', interpolation='bilinear', cmap='viridis')
        plt.scatter(x[greens==1], y[greens==1], color='green', s=20, marker = '^')
        plt.scatter(x[reds==1], y[reds==1], color='red', s=20, marker = 'o')
        plt.scatter(x[blues==1], y[blues==1], color='blue', s=20, marker = 's')

        for i in data['labels'][str(idx)]:
            if i[4] == 0:
                color = 'blue'
            elif i[4] == 1:
                color = 'red'
            elif i[4] == 2:
                color = 'green'
            else:
                continue

            x = np.array([int(0.5 * (i[1] + i[3]))])
            y = np.array([int(0.5 * (i[0] + i[2]))])
            b = return_static_box_bounds(x, y, dmap, STANDARD_BOXSIZE_X, STANDARD_BOXSIZE_Y)[0]

            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        plt.draw()
        plt.pause(delay)

classification_demo(1)