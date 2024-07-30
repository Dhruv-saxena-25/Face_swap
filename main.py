import cv2
import numpy as np
import dlib
import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("Face Swapper APP")

col1, col2 =  st.columns([1, 2])

with col1:
    image = st.file_uploader("Upload Source Image")
    if image:
        image_path = os.path.join(image.name)
        img = Image.open(image_path)
        st.image(img)
with col2:
    st.title("Webcam Live Feed")
    FRAME_WINDOW = st.image([])

if st.button("Swap"):
    #select source image "with whole face visible"(prefer lips closed)
    img = cv2.imread(image_path)
    # if(img is not None):
    #     gray1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray1)

    #create a face detector and a landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    indices = []

    #Extracting face from source
    faces = detector(gray1)
    for face in faces:
        landmarks = predictor(gray1, face)
        #create an array to add coordinates of the landmarks
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        points = np.array(landmarks_points, np.int32)
        #creating a convex polygon covering all the points
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)
        #extracting face(the convexhull formed) out of the image
        face_image_1 = cv2.bitwise_and(img, img, mask=mask)
        
        #Delaunay triangulation(form triangles with the points)
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        #for each point(x,y) in a triangle, replace it by its index(0-68) from the 68 landmarks 
        indices = []
        def index_nparray(nparray):
            index = None
            for num in nparray[0]:
                index = num
                break
            return index 
        
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = index_nparray(index_pt3)


            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indices.append(triangle)

    #select destination video(single person video is preferable)
    capture = cv2.VideoCapture(0)#param=0 for webcam.
    #If webcam is used, use a voice recorder to record and add it in the later part.
    capture.set(3, 1000)
    capture.set(4, 550)

    #create a VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #adjust the 3rd parameter incase the output video is not the same size as of original video. 
    out = cv2.VideoWriter('output_noaudio.mp4', fourcc,9.5, (640,480))

    #start processing each frame in the video till the end
    cread=capture.read()
    while cread!=(False,None):
        img2 = cread[1]
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2_new_face = np.zeros_like(img2)

        #detecting a face in the video and predicting the landmarks on it 
        faces2 = detector(gray2)
        for face in faces2:
            landmarks = predictor(gray2, face)
            #create an array to add the coordinates of all the landmarks
            landmarks_points2 = [] 
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points2.append((x, y))
            points2 = np.array(landmarks_points2, np.int32)
            #create a convex polygon containing all the points
            convexhull2 = cv2.convexHull(points2)
        #create an array of 0s(a black frame) for both image and video frame
        lines_space_mask = np.zeros_like(gray1)
        lines_space_new_face = np.zeros_like(img2)
        
    #Divide the face in both image and video into triangles with the indices taken earlier
        for triangle_index in indices:
            # Triangulation of the first face
            point1 = landmarks_points[triangle_index[0]]
            point2 = landmarks_points[triangle_index[1]]
            point3 = landmarks_points[triangle_index[2]]
            triangle1 = np.array([point1,point2,point3], np.int32)
            
            rect1 = cv2.boundingRect(triangle1)
            (x, y, w, h) = rect1
            cropped_triangle = img[y: y + h, x: x + w]
            cropped_tr1_mask = np.zeros((h, w), np.uint8)
            
            #create a 2x3 matrix of coordinates of each point to apply Affine transformation to it 
            points = np.array([[point1[0] - x, point1[1] - y],
                            [point2[0] - x, point2[1] - y],
                            [point3[0] - x, point3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
            
            # Triangulation of second face
            point1 = landmarks_points2[triangle_index[0]]
            point2 = landmarks_points2[triangle_index[1]]
            point3 = landmarks_points2[triangle_index[2]]
            triangle2 = np.array([point1, point2, point3], np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x, y, w, h) = rect2
            cropped_tr2_mask = np.zeros((h, w), np.uint8)

            #create a 2x3 matrix of coordinates of each point to apply Affine transformation to it
            
            points2 = np.array([[point1[0] - x, point1[1] - y],
                                [point2[0] - x, point2[1] - y],
                                [point3[0] - x, point3[1] - y]], np.int32)
            cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

            #Change the dimensions of each triangle in source image to match
            #the dimensions of corresponding triangles in destination face
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

            # Reconstructing source face in the place of destination face in video
            img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
            img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
            img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
            img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
           
        # Replace destination face by source face)
        img2_face_mask = np.zeros_like(gray2)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)
        img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)
        frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        cv2.imshow("result", frame)
        
        #Now run seamlessClone to fit the face in destination video 
        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w)/2),int((y + y + h)/2))
        seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)
        #save it to output file
        out.write(seamlessclone)
        cread=capture.read()
        key = cv2.waitKey(1)
        if key == 27:#hit "esc" to close
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()

