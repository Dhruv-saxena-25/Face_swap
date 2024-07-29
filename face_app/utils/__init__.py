import sys
import cv2
import numpy as np


from face_app.exception import FaceException




def index_nparray(nparray):
    try:
        index = None
        for num in nparray[0]:
            index = num
            break
        return index
    except Exception as e:
        raise FaceException(e, sys)

def transformation(triangle_index, img, points_list):
    try:
        tr_pt1 = points_list[triangle_index[0]]
        tr_pt2 = points_list[triangle_index[1]]
        tr_pt3 = points_list[triangle_index[2]]

        triangle = np.array([tr_pt1, tr_pt2, tr_pt3], np.int32)

        rect = cv2.boundingRect(triangle)

        (x, y, w, h) = rect

        cropped_triangle = img[y: y + h, x: x + w]

        cropped_tr_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr_pt1[0] - x, tr_pt1[1] - y],
                        [tr_pt2[0] - x, tr_pt2[1] - y],
                        [tr_pt3[0] - x, tr_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr_mask, points, 255)

        cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                        mask=cropped_tr_mask)

        return cropped_triangle, points, cropped_tr_mask, rect
    except Exception as e:
        raise FaceException(e, sys)

def save_img(filepath,image):
    try:
        cv2.imwrite(filepath, image)
    except Exception as e:
        raise FaceException(e, sys)