from pathlib import Path
import pickle
import cv2
import numpy as np
import pandas as pd
import uuid
import time
from collections import deque, Counter

CASCADE_PATH = Path('./src/model/haarcascade_frontalface_default.xml')  # Path to Haar Cascade model
GENDER_MODEL_PATH = Path('./src/model/svm_gender.pickle')  # Path to gender SVM model
IDENTITY_MODEL_PATH = Path('./src/model/svm_identity.pickle')  # Path to identity SVM model
PCA_MODEL_PATH = Path('./src/model/pca_dict.pickle')  # Path to PCA model

# Load Haar Cascade model for face detection
haar = cv2.CascadeClassifier(str(CASCADE_PATH))  # Load Haar Cascade model
with open(GENDER_MODEL_PATH, 'rb') as f:  # Load gender SVM model
    genderModel_svm = pickle.load(f)
with open(IDENTITY_MODEL_PATH, 'rb') as f:  # Load identity SVM model
    identityModel_svm = pickle.load(f)
with open(PCA_MODEL_PATH, 'rb') as f:  # Load PCA model
    pca_models = pickle.load(f)

model_pca = pca_models['pca']  # Extract PCA model
mean_face = pca_models['mean_face']  # Extract mean face

# Identity to gender mapping
data = np.load('./src/data/pca_data_50_target.npz', allow_pickle=True)
identities = data['identity']
genders = data['gender']
identity_to_gender = pd.DataFrame({'identity': identities, 'gender': genders}).drop_duplicates(
    'identity').set_index('identity')['gender'].to_dict()

class FaceTrack:
    def __init__(self, max_distance=50):
        self.tracked_faces = {}
        self.max_distance = max_distance

    def _find_closest_face(self, x, y):
        min_dist = float('inf')
        matched_id = None
        for fid, data in self.tracked_faces.items():
            prev_x, prev_y = data['position']
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            if dist < self.max_distance and dist < min_dist:
                matched_id = fid
                min_dist = dist
        return matched_id
    
    def update(self, x, y, identity):
        now = time.time()
        face_id = self._find_closest_face(x,y)

        if face_id is None:
            face_id = str(uuid.uuid4())

        if face_id not in self.tracked_faces:
            self.tracked_faces[face_id] = {
                'buffer': deque(maxlen=60),
                'start_time': now,
                'last_seen': now,
                'fixed_identity': None,
                'position': (x,y)
            }

        track = self.tracked_faces[face_id]
        track['buffer'].append(identity)
        track['last_seen'] = now
        track['position'] = (x,y)

        if track['fixed_identity'] is None and (now - track['start_time']) >= 1:
            most_common = Counter(track['buffer']).most_common(1)[0][0]
            track['fixed_identity'] = most_common

        return track['fixed_identity'] if track['fixed_identity'] else 'Identifying...'
    
    def cleanup(self, timeout=1.0):
        now = time.time()
        to_delete = [fid for fid, val in self.tracked_faces.items() if now - val['last_seen'] > timeout]
        for fid in to_delete:
            del self.tracked_faces[fid]

face_tracker = FaceTrack()

def draw_stylish_box(img, x, y, w, h, label, color):
    # Draw corner lines for a modern box look
    line_length = 30
    thickness = 2

    # Top-left
    cv2.line(img, (x, y), (x + line_length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_length), color, thickness)

    # Top-right
    cv2.line(img, (x + w, y), (x + w - line_length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + line_length), color, thickness)

    # Bottom-left
    cv2.line(img, (x, y + h), (x, y + h - line_length), color, thickness)
    cv2.line(img, (x, y + h), (x + line_length, y + h), color, thickness)

    # Bottom-right
    cv2.line(img, (x + w, y + h), (x + w - line_length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_length), color, thickness)

    # Label background
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)

    # Label text
    cv2.putText(img, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def faceRecognitionPipeline(filename, path: bool = True):
    img = cv2.imread(str(filename)) if path else filename  # Read image from file or use provided image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))  # Detect faces in the image

    predictions = []  # Initialize list to store predictions

    for i, (x, y, w, h) in enumerate(faces):  # Iterate over detected faces
        roi = gray[y:y + h, x:x + w] / 255.0 # Extract region of interest (face) and normalize pixel values
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)  # Resize face to 100x100 pixels
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
        
        roi_reshape = roi_resize.reshape(1, 10000) # Flatten the resized face image
        roi_mean = roi_reshape - mean_face  # Center the face image by subtracting the mean face
        eigen_image = model_pca.transform(roi_mean)  # Apply PCA transformation
        eig_img = model_pca.inverse_transform(eigen_image) # For visualization purposes
        
        pred_identity = identityModel_svm.predict(eigen_image)[0]  # Predict identity
        identity_proba = identityModel_svm.predict_proba(eigen_image).max() # Get probability
       
        
        
        if identity_proba < 0.6 or pred_identity.lower() == 'unknown':
            pred_identity = 'unknown'
            pred_gender = genderModel_svm.predict(eigen_image)[0]  # Predict
            gender_proba = genderModel_svm.predict_proba(eigen_image).max() # Get probability
        else:
            pred_gender = identity_to_gender.get(pred_identity, 'unknown')
            gender_proba = identity_proba

        stable_identity = face_tracker.update(x + w // 2, y + h // 2, pred_identity)

        if stable_identity == 'unknown':
            text = f'Unknown ({pred_gender})'
        elif stable_identity == 'Identifying...':
            text = f'Identifying... ({pred_gender})'
        else:
            text = f'{stable_identity} ({pred_gender})'

        # Bounding box around the face by gender
        color = (255, 0, 255) if pred_gender == 'female' else (255, 255, 0)
        draw_stylish_box(img, x, y, w, h, text, color)

        predictions.append({
            'roi': roi_resize,
            'eig_img': eig_img,
            'gender': pred_gender,
            'identity': pred_identity,
            'score_gender': gender_proba,
            'score_identity': identity_proba
        })

    face_tracker.cleanup()
    return img, predictions  # Return the annotated image and predictions

