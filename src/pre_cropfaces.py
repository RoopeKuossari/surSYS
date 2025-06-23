import numpy as np # numerical python
import cv2 # opencv
from glob import glob # glob use to extract path of file
import matplotlib.pyplot as plt # visualze and display 
import os # to create directory

# Path for unknown faces
fupath = glob('./data/images/train/female/unknown/*.jpg')
mupath = glob('./data/images/train/male/unknown/*.jpg')

print('The number of images in unknown female folder = ',len(fupath))
print('The number of images in unknown male folder = ',len(mupath))

haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml') # load haar cascade classifier

for i in range(len(fupath)):
    try:
            
        #Read Image and Convert to RGB
        img = cv2.imread(fupath[i])
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #Apply Haar Cascade Classifier
        gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
        faces_list = haar.detectMultiScale(gray,1.5,5)
        for x,y,w,h in faces_list:
            #crop Face 
            roi = img[y:y+h,x:x+w]
            #Save Image
            cv2.imwrite(f'./data/images/crop/female/unknown/female_{i}.jpg',roi)
            print('Image Sucessfully processed')
    except:
        print('Unable to Process the image')

for i in range(len(mupath)):
    try:

        #Read Image and Convert to RGB
        img = cv2.imread(mupath[i])
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        #Apply Haar Cascade Classifier
        gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
        faces_list = haar.detectMultiScale(gray,1.5,5)
        for x,y,w,h in faces_list:
            #crop Face
            roi = img[y:y+h,x:x+w]
            #Save Image
            cv2.imwrite(f'./data/images/crop/male/known/male_{i}.jpg',roi)
            print('Image Sucessfully processed')
    except:
        print('Unable to Process the image')



def crop_and_save_faces(src_root, dest_root):

    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_root)
                dest_path = os.path.join(dest_root, rel_path)

                # Create destination folder if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Read and process image
                img = cv2.imread(src_path)
                img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # this step will convert image from BGR to RGB
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                faces_list = haar.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                # Crop and save all detected faces (can be adjusted to just take the largest face, etc.)
                for i, (x, y, w, h) in enumerate(faces_list):
                    roi = img[y:y+h, x:x+w]
                    if len(faces_list) > 1:
                        filename, ext = os.path.splitext(file)
                        face_filename = f"{filename}_face{i}{ext}"
                    else:
                        face_filename = file
                    face_path = os.path.join(os.path.dirname(dest_path), face_filename)
                    cv2.imwrite(face_path, roi)

                if len(faces_list) == 0:
                    print(f"No face found in: {src_path}")

# Example usage
src_known = "./data/images/train/female/known"
dst_known = "./data/images/crop/female/known"
crop_and_save_faces(src_known, dst_known)
src_known = "./data/images/train/male/known"
dst_known = "./data/images/crop/male/known"
crop_and_save_faces(src_known, dst_known)