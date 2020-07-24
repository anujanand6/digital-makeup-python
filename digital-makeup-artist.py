
#* Title: Python code to apply digital makeup on an image of a person 

# import required packages
import imutils
#*    Title: imutils
#*    Author: Adrian Rosebrock
#*    Date: August 18, 2019
#*    URL: https://github.com/jrosebr1/imutil

from imutils import face_utils

import dlib
#*    Title: dlib
#*    Author: Davis E. King
#*    Year: 2002
#*    URL: https://github.com/davisking/dlib

import cv2

from PIL import Image, ImageDraw


# Detect face using dlib using HOG + Linear SVM detector
face_detector = dlib.get_frontal_face_detector()

# Predict the facial landmark using ensemble regression trees
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load the input image
image = cv2.imread("input_image.jpg")   

# Resize the image
image = imutils.resize(image, width=500)

# Convert the image from BGR to RGB
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert original image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
box = face_detector(image_RGB, 1)

image_pil = Image.fromarray(image_RGB)

# Create a PIL object to draw on the image
makeup = ImageDraw.Draw(image_pil, 'RGBA')

for (i, box) in enumerate(box):
	# Predict the landmarks features
    points = landmark_predictor(gray, box)
    points = face_utils.shape_to_np(points)
    
    # Convert shape to tuple
    points = tuple(map(tuple, points))
    
    # Obtain list of face landmarks with the respective 68 facial coordinate points
    landmarks_list = face_utils.FACIAL_LANDMARKS_IDXS.items()
    
    # Iterate over the landmark list
    for (face_landmarks, (i,j)) in landmarks_list:
              
        # Start applying makeup on the corresponding locations
        
        # Highlight the eyebrows
        if face_landmarks == "left_eyebrow":
            makeup.polygon(points[i:j], fill=(68, 54, 39, 128))
            makeup.line(points[i:j], fill=(68, 54, 39, 150), width=5)
        
        if face_landmarks == "right_eyebrow":
            makeup.polygon(points[i:j], fill=(68, 54, 39, 128))
            makeup.line(points[i:j], fill=(68, 54, 39, 150), width=5)
        
        # Add lipstick
        if face_landmarks == "mouth":
            makeup.polygon(points[i:j], fill=(150, 0, 0, 128))
            makeup.polygon(points[i:j], fill=(150, 0, 0, 128))
            makeup.line(points[i:j], fill=(150, 0, 0, 64), width=8)
            makeup.line(points[i:j], fill=(150, 0, 0, 64), width=8)
        
        # Apply eyeliner
        if face_landmarks == "left_eye":
            makeup.line(points[i:j] + points[i:j][0], fill=(0, 0, 0, 110), width=6)
    
        if face_landmarks == "right_eye":
            makeup.line(points[i:j] + points[i:j][0], fill=(0, 0, 0, 110), width=6)
            
    # Show and save the image in the working directory       
    image_pil.show()
#image_pil.save("output\output_image.jpg")