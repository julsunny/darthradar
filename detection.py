"""
Detection of regions of interests in the radar doppler maps
"""

from typing import Union, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from dataloader import RadarImageTargetSet

# type alias for boxes
Box = np.ndarray

def IOU(box1: Box, box2: Box) -> float:
    """IOU = Area of Union / Area of Intersection
    
    Returns:
    --------
        int: The quality of the match
            1 in case of a perfect match
            0 in case of no overlap of boxes
    """

    if len(box1) == 5:
        box1 = box1[:-1]
    if len(box2) == 5:
        box2 = box2[:-1]

    # calculate intersection area
    axmin, aymin, axmax, aymax = box1
    bxmin, bymin, bxmax, bymax = box2

    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    if (dx>=0) and (dy>=0):
        I = dx*dy
    else:
        return 0
    
    # union = area1 + area2 - intersection
    U =  (axmax - axmin) * (aymax - aymin) + (bxmax - bxmin) * (bymax - bymin) - I 

    return I / U



def evalute_box_finder(peak_detector, data_set: RadarImageTargetSet):
    """
    Evaluates performance of a peak detector on a dataset.

    Arguments:
    ----------
    peak_detector: 2d np.array -> FloatTensor[N, 4]
        a function that takes an image as a 2d numpy array and returns an N x 4 matrix where
        N is the number of boxes / peaks found
        each row contains x0, y0, x1, y1
        Example:
        peak_detector(img) gives
            [[1, 5, 2, 3]
            ,[7, 8, 1, 3]
            ]]
            (N = 2)
    """
    assert len(data_set) > 0
    assert isinstance(data_set[0][0], np.ndarray)
    assert all(key in data_set[0][1] for key in ['boxes', 'labels', 'image_id', 'area'])
    assert isinstance(data_set[0][1]['boxes'], np.ndarray)
    assert data_set[0][1]['boxes'].shape[1] == 4
    # some more asserts maybe?

    box_errors: Dict[int, Union[float, str]] = dict()
    for (img, tgt) in data_set:
        found_boxes = peak_detector(img)
        evaluate_boxes_pair(found_boxes, tgt['boxes'])

def evaluate_boxes_pair(boxes1, boxes2) -> Union[float, str]:
    """Evaluate a pair of box labels"""
    if len(boxes1) != len(boxes2):
        return 'mismatched number of boxes'

    # match found boxes to labels
    goodness_matrix = np.array([[IOU(label_box, found_box) for label_box in boxes1] for found_box in boxes2])
    print("goodness matrix:\n", goodness_matrix)
    row_indices, col_indices = linear_sum_assignment(goodness_matrix, maximize=True)
    # calculate the sum of the matched pairs' goodness
    print("idxs:", row_indices, col_indices)
    print("goodness:", [goodness_matrix[row_idx, col_idx] for row_idx, col_idx in zip(row_indices, col_indices)])
    total_goodness = sum(goodness_matrix[row_idx, col_idx] for row_idx, col_idx in zip(row_indices, col_indices))

    # return goodness per box
    return total_goodness / len(boxes1)

#
# TESTING CODE, don't use
#

def test_evaluate_boxes_trivial():
    """Test the evaluate_box_function by giving it the same set of boxes twice""" 
    test_boxes = np.array([[  5., 110.,  10., 116.,   2.], [  4., 108.,  12., 116.,   2.]])
    assert np.isclose(1., evaluate_boxes_pair(test_boxes, test_boxes))