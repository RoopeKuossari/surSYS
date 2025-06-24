from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Collect paths and create DataFrame
DATA_PATH = Path('./data/data_images_100_100.pickle')
MODEL_PATH = Path('./model/pca_dict.pickle')
PCA_DATA_PATH = Path('./data/pca_data_50_target.npz')

# Function to compute PCA on the dataset
def compute_pca() -> None:
    # Load the preprocessed data
    data = pickle.load(open(DATA_PATH, 'rb'))
    X = data.drop(['gender', 'identity'], axis=1).values # Extract features
    y_gender = data['gender'].values # Extract gender labels
    y_identity = data['identity'].values # Extract identity labels

    mean_face = X.mean(axis=0) # Compute the mean face
    X_t = X - mean_face  # Center the data by subtracting the mean face
    
    pca = PCA(n_components=None, whiten=True, svd_solver='auto') # Initialize PCA
    pca.fit(X_t) # Fit PCA to the centered data

    exp_var_df = pd.DataFrame({ 
        'explained_var': pca.explained_variance_ratio_, # Explained variance ratio
        'cum_explained_var': np.cumsum(pca.explained_variance_ratio_), # Cumulative explained variance
        'principal_components': np.arange(1, len(pca.explained_variance_ratio_) + 1), # Principal components
    }).set_index('principal_components')

    plt.figure() # Plot explained variance
    exp_var_df['cum_explained_var'].plot(title='Cumulative Explained Variance') # Plot cumulative explained variance
    plt.xlabel('Number of components') # Set x-axis label
    plt.ylabel('Cumulative Explained Variance') # Set y-axis label
    plt.savefig('./data/pca_variance.png') # Save the plot
    plt.close() # Close the plot

    pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto') # Initialize PCA with 50 components
    pca_data = pca_50.fit_transform(X_t) # Fit PCA to the centered data and transform it

    np.savez(PCA_DATA_PATH, X=pca_data, gender=y_gender, identity=y_identity) # Save PCA data to a .npz file

    pca_dict = { # Store PCA model and mean face in a dictionary
        'pca': pca_50,
        'mean_face': mean_face
    }

    with open(MODEL_PATH, 'wb') as f: # Save the PCA model and mean face to a pickle file
        pickle.dump(pca_dict, f) # Save the PCA model and mean face

if __name__ == "__main__": # Main entry point
    compute_pca()  # Run PCA computation
    print("PCA computation completed and models saved.")