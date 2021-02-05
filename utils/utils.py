"""
Code by Muhammad Nouman Ahsan(Original Author)

Differnt Faces Recognition in Videos
IDE: VS Code(you can use it in other IDEs)

"""

import os
import cv2
import numpy as np
import datetime


# creating folders that are necessary
def _create_structure(output_directory, video_name):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists(os.path.join(output_directory, video_name, "images")):
        os.makedirs(os.path.join(output_directory, video_name, "images"))


# locations needed while croping face from the image
def _new_locations(point, _frame_dim):
    l = abs(point[1] - point[0])
    _decre = int(l / 4)
    while True:
        if point[0] - l > 0 and point[1] + l < _frame_dim:
            p1 = point[0] - l
            p2 = point[1] + l
            break
        else:
            l = l - _decre if l > 0 else 0
    return (p1, p2)


# crop face from frame
def crop_face(frame, bbox):
    _frame_height, _frame_width = frame.shape[0], frame.shape[1]
    top, right, bottom, left = bbox
 
    _top, _bottom = _new_locations((top, bottom), _frame_height)
    _left, _right = _new_locations((left, right), _frame_width)
    return frame[_top:_bottom, _left:_right]


# take average of face data
def _take_average(_encodings):
    _array = np.array(_encodings)
    return np.mean(_array, axis=0)
