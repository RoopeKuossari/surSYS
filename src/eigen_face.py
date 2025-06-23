import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import pickle

# Load the preprocessed data
data = pickle.load(open('./data/data_images_100_100.pickle', mode='rb'))

# Extract features and labels
X = data.drop(['gender', 'identity'], axis=1).values
y_gender = data['gender'].values
y_identity = data['identity'].values

# Calculate the mean face
mean_face = X.mean(axis=0)
# Center the data by subtracting the mean face
X_t = X - mean_face
# Perform PCA on the centered data
# Initialize PCA with whiten=True to normalize the components
# and svd_solver='auto' to let sklearn choose the best method
pca = PCA(n_components=None, whiten=True, svd_solver='auto')
pca.fit(X_t)

# Transform the data using PCA
exp_var_df = pd.DataFrame({
    'explained_var': pca.explained_variance_ratio_,
    'cum_explained_var': np.cumsum(pca.explained_variance_ratio_),
    'principal_components': np.arange(1, len(pca.explained_variance_ratio_) + 1)
}).set_index('principal_components')

# Plot the explained variance
pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto')
pca_data = pca_50.fit_transform(X_t)

# Save the PCA-transformed data and the corresponding labels
np.savez('./data/pca_data_50_target.npz', X=pca_data, gender=y_gender, identity=y_identity)
pca_dict = {
    'pca': pca_50,
    'mean_face': mean_face
}
pickle.dump(pca_dict, open('./model/pca_dict.pickle', mode='wb'))