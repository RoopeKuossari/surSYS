from pathlib import Path
import pickle
import cv2
import numpy as np

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

def faceRecognitionPipeline(filename, path: bool = True):
    img = cv2.imread(str(filename)) if path else filename  # Read image from file or use provided image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = haar.detectMultiScale(gray, 1.5, 3)  # Detect faces in the image
    predictions = []  # Initialize list to store predictions
    for (x, y, w, h) in faces:  # Iterate over detected faces
        roi = gray[y:y + h, x:x + w]  # Extract region of interest (face)
        roi = roi / 255.0  # Normalize pixel values
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)  # Resize face to 100x100 pixels
        else:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_CUBIC)
        roi_reshape = roi_resize.reshape(1, 10000) # Flatten the resized face image
        roi_mean = roi_reshape - mean_face  # Center the face image by subtracting the mean face
        eigen_image = model_pca.transform(roi_mean)  # Apply PCA transformation
        eig_img = model_pca.inverse_transform(eigen_image) # For visualization purposes
        pred_gender = genderModel_svm.predict(eigen_image)[0]  # Predict
        gender_proba = genderModel_svm.predict_proba(eigen_image).max() # Get probability
        pred_identity = identityModel_svm.predict(eigen_image)[0]  # Predict identity
        identity_proba = identityModel_svm.predict_proba(eigen_image).max() # Get probability
        if identity_proba < 0.6:
            pred_identity = 'unknown'
        if pred_identity.lower() == 'unknown':
            text = f"Unknown ({pred_gender}): {int(gender_proba * 100)}%"
        else:
            text = f"{pred_identity} ({pred_gender}): {int(max(identity_proba, gender_proba) * 100)}%"

        # Generate report
        print(text)

        # Bounding box around the face by gender
        color = (255, 0, 255) if pred_gender == 'female' else (255, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        predictions.append({
            'roi': roi_resize,
            'eig_img': eig_img,
            'gender': pred_gender,
            'identity': pred_identity,
            'score_gender': gender_proba,
            'score_identity': identity_proba
        })

    return img, predictions  # Return the annotated image and predictions

