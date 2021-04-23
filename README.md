# Infineon Radar Challenge

## Introduction

## Data Format

The data is provided in the HDF5 format. It has the following hierachy:

<pre>
data.h5
|-- rdms
|   |-- 3D array (374 x 256 x 32)
|-- labels
|   |-- '0'
|   |   |-- 2D array (n x 5)
|   |-- '1'
|   |   |-- 2D array (n x 5)
|   |-- ...
|   |   |-- 2D array (n x 5)
|   |-- '374'
|   |   |-- 2D array (n x 5)
</pre>

### Range-Doppler Map
The signal from the radar sensor is represented as range-Doppler map. The Doppler map is a 2D array with a dimension of 256 x 32. Since there are 374 Doppler map samples they are stored as 3D array with 374 samples x 256 width x 32 height. This data can be used as starting point for feature generation.

### Labels
The corresponding labels are 2D arrays with dimension n x 5 with n being the number of objects detected in the range Doppler map. These arrays are stored as fields in the `labels` dataset. The fields are named by the sample number e.g. `'0'`, `'1'`. The 5 elements of each label row contain the following information:

<pre>
labels
|-- '0'
|   |-- xmin, ymin, xmax, ymax, class_type
|   |-- xmin, ymin, xmax, ymax, class_type
|-- '1'
|   |-- xmin, ymin, xmax, ymax, class_type
|-- ...
</pre>


<pre>
0: pedestrian
1: car
2: truck
3: no object (like no entry)
</pre>

## Getting Started (Python)

### Requirements
It is recommended to use Python 3.9. Please check [here](https://www.python.org/downloads/) how to install it on your system.
Further, the following package is required:
- h5py (to load the HDF data file)

Additionally, it is recommended to install these packages:
- pandas (helps to represent the data in the correct format)
- sklearn (machine learning framework)
- matplotlib (for generating plots)

These packages can be installed using `pip`:
```
pip3 install h5py pandas sklearn matplotlib
```

If `pip` is not installed on your system, you can install it by typing
```
python3 -m ensurepip
```

### Coding Examples

#### clf_example.py
The example `clf_example.py` example reads the range Doppler image (256 x 32 matrix) and converts it to a vector with 8192 elements. This vector is used as feature vector X. The value y we want to learn in this simple example is the number of targets within a certain radar frame. Because of the low amount of potential outcomes this can be seen as classification problem. For demonstration the MLP Classifier of the Sklearn library is used and a confusion matrix is plotted.

Please open the file `clf_example.py` and read the comments to understand how to work with the dataset.

#### plotDopplerMap.py
This example plots the range Doppler map as colored image and shows positions
and types of the different labels within this image. This example should help
to get a better understanding about the dataset.

Since there are a lot of sample this example will run for a while. Please use CTRL-C in the command line (you may need to spam it) to exit the example.