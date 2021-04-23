"""
Detection of regions of interests in the radar doppler maps
"""

from typing import Dict

import h5py
import torch
import numpy as np
from scipy import optimize.linear_sum_assignment

Box = torch.Tensor

def IOU(box1: Box, box2: Box) -> f32:
    """IOU = Area of Union / Area of Intersection"""
    # WARNING this function uses lower left corner, width, height to define a box
    x0, y0, x1, y1 = box1
    x1, y1, w1, h1 = x0, y0, x1 - x0, y1 - y0

    x0, y0, x1, y1 = box2
    x2, y2, w2, h2 = x0, y0, x1 - x0, y1 - y0

    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)

    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)

    if w_intersection <= 0 or h_intersection <= 0: 
        # area of intersection is 0 -> cost is infinite
        return float(inf)

    I = w_intersection * h_intersection

    U = w1 * h1 + w2 * h2 - I 

    return I / U

def evalute_box_finder(peak_detector, data_set: torch.utils.data.Dataset):
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
    assert isinstance(data_set[0][1]['boxes'], torch.FloatTensor)
    # some more asserts maybe?

    box_errors: Dict[int, Union[float, str]] = dict()
    for (img, tgt) in data_set:
        found_boxes = peak_detector(img)

        if len(found_boxes) != len(tgt['boxes']):
            box_errors[tgt['image_id']] = 'mismatched number of boxes'
        
        # match found boxes to labels
        cost_matrix = np.array([[IOU(label_box, found_box) for label_box in tgt['boxes']] for found_box in found_boxes])
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        # calculate the sum of the matched pairs' costs
        total_cost = sum(cost_matrix[row_idx, col_idx] for row_idx, col_idx in zip(row_indices, col_indices))
        box_errors[tgt['image_id']] = total_cost







#iou = [IOU(y_test[i], y_pred[i]) for i in range(len(x_test))]