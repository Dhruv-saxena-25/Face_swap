import dlib
import cv2
import numpy as np

import logging
import os
import sys

from face_app.exception import FaceException
from face_app.utils import index_nparray, transformation, save_img

logger = logging.getLogger(__name__)

class FaceAPP():
    
    def __init__(self, source_filepath, destination_filepath,  output_filepath):
        try:
            self.source_filepath = source_filepath
            self.destination_filepath = destination_filepath
            self.output_filepath = output_filepath

            self.face_detector =  dlib.get_frontal_face_detector()
            self.points_detector =  dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
            
            logger.info("FaceApp instance created")
        except Exception as e:
            raise FaceException(e, sys) from e

    def read_source_image(self):

        try:
            
            self.source = cv2.imread(self.source_filepath)
            source_grey = cv2.cvtColor(self.source, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(source_grey)
            source_faces = self.face_detector(source_grey)

            logger.info('reading source image')
            return source_grey , mask, source_faces
        except Exception as e:
            raise FaceException(e, sys) from e    


    def create_contour(self,  source_grey, mask, source_faces):
        try:
            for face in source_faces:
                points = self.points_detector(source_grey, face)
                source_points_list = []
                for n in range(0, 68):
                    x = points.part(n).x
                    y = points.part(n).y
                    source_points_list+=[(x, y)]

                source_points = np.array(source_points_list, np.int32)
                convexhull = cv2.convexHull(source_points)
                cv2.fillConvexPoly(mask, convexhull, 255)
                source_face = cv2.bitwise_and(self.source, self.source, mask=mask)
            logger.info('Contour Created')

            return source_points_list, convexhull, source_points

        except Exception as e:
            raise FaceException(e, sys) from e  

    def extract_triangles(self,  points_list, convexhull):
        try:
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(points_list)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)
            return triangles

        except Exception as e:
            raise FaceException(e, sys) from e  
        
    def create_triangle_id(self, triangles, points):
        try:
            triangles_id = []
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                id_pt1 = np.where((points == pt1).all(axis=1))
                id_pt1 = index_nparray(id_pt1)
                id_pt2 = np.where((points == pt2).all(axis=1))
                id_pt2 = index_nparray(id_pt2)
                id_pt3 = np.where((points == pt3).all(axis=1))
                id_pt3 = index_nparray(id_pt3)

                if id_pt1 is not None and id_pt2 is not None and id_pt3 is not None:
                    triangle = [id_pt1, id_pt2, id_pt3]
                    triangles_id.append(triangle)

            logger.info('triangle id created')
            
            return triangles_id

        except Exception as e:
            raise FaceException(e, sys) from e


    def read_destination_image(self):

        try:
            
            self.destination = cv2.imread(self.destination_filepath)
            destination_grey = cv2.cvtColor(self.destination, cv2.COLOR_BGR2GRAY)
            destination_faces = self.face_detector(destination_grey)

            logger.info('Extraction of source image pints started')

            for face in destination_faces:
                points_predict2 = self.points_detector(destination_grey, face)
                destination_points_list = []
                for n in range(0, 68):
                    x = points_predict2.part(n).x
                    y = points_predict2.part(n).y
                    destination_points_list.append((x, y))
                points = np.array(destination_points_list, np.int32)
                destination_hull = cv2.convexHull(points)
            logger.info('Extraction finished')

            return destination_grey, destination_hull, destination_points_list

        except Exception as e:
            raise FaceException(e, sys) from e
        
    
    def fit(self, triangles_id, source_point_list, destination_points_list):
        try:
            logger.info('Face mask creating started')
            destination_new_face = np.zeros_like(self.destination, np.uint8)
            for triangle_index in triangles_id:
            
                cropped_triangle, points, cropped_tr1_mask, rect1 = transformation(triangle_index, self.source, source_point_list)

                cropped_triangle2, points2, cropped_tr2_mask, rect2 = transformation(triangle_index, self.destination, destination_points_list)

                (x2, y2, w2, h2) = rect2

                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)


                img2_new_face_rect_area = destination_new_face[y2: y2 + h2, x2: x2 + w2]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                destination_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area
            
            logger.info('Face Mask Created')

            return destination_new_face

        except Exception as e:
            raise FaceException(e, sys) from e
        
    
    def render_face(self, destination_grey, destination_hull, destination_new_face):
        try:
            logger.info('Rendering generated image')

            destination_face_mask = np.zeros_like(destination_grey)
            destination_head_mask = cv2.fillConvexPoly(destination_face_mask, destination_hull, 255)
            destination_face_mask = cv2.bitwise_not(destination_head_mask)

            destination_noface = cv2.bitwise_and(self.destination, self.destination, mask=destination_face_mask)


            result = cv2.add(destination_noface, destination_new_face)

            #cloning face into the img2
            (x3, y3, w3, h3) = cv2.boundingRect(destination_hull)
            center_face = (int((x3 + x3 + w3) / 2), int((y3 + y3 + h3) / 2))


            clone = cv2.seamlessClone(result, self.destination, destination_head_mask, center_face, cv2.MIXED_CLONE)
        
        
            save_img(os.path.join(os.getcwd(), self.output_filepath), clone)
            logger.info('rendering finished')
        except Exception as e:
            raise FaceException(e, sys) from e
    

    def run(self):
        try:
            source_grey , mask, source_faces = self.read_source_image()

            source_points_list, convexhull, source_points = self.create_contour(source_grey, mask, source_faces)

            triangles = self.extract_triangles(source_points_list, convexhull)

            triangles_id = self.create_triangle_id(triangles, source_points)

            destination_grey, destination_hull, destination_points_list = self.read_destination_image()

            destination_new_face = self.fit(triangles_id, source_points_list, destination_points_list)

            self.render_face(destination_grey, destination_hull, destination_new_face)

        except Exception as e:
            raise FaceException(e, sys)
