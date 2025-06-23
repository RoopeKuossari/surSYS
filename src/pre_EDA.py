import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# computer vision library
import cv2

# glob
from glob import glob
import pickle

# Get all female and male cropped image paths
fpath = (
    glob('./src/data/images/crop/female/**/*.jpg', recursive=True) +
    glob('./src/data/images/crop/female/**/*.jpeg', recursive=True) +
    glob('./src/data/images/crop/female/**/*.png', recursive=True)
)
mpath = (
    glob('./src/data/images/crop/male/**/*.jpg', recursive=True) +
    glob('./src/data/images/crop/male/**/*.jpeg', recursive=True) +
    glob('./src/data/images/crop/male/**/*.png', recursive=True)
)

# Extract identity from the file path
def extract_identity(path):
    parts = path.replace("\\", "/").split('/')
    if 'known' in parts:
        return parts[parts.index('known') + 1]  # e.g., "julia" or "roope"
    elif 'unknown' in parts:
        return 'unknown'
    return 'unknown'  # fallback

# Create DataFrames for each gender
df_female = pd.DataFrame(fpath, columns=['filepath'])
df_female['gender'] = 'female'
df_female['identity'] = df_female['filepath'].apply(extract_identity)
df_male = pd.DataFrame(mpath, columns=['filepath'])
df_male['gender'] = 'male'
df_male['identity'] = df_male['filepath'].apply(extract_identity)
# Combine both DataFrames
df = pd.concat((df_female, df_male), axis=0, ignore_index=True)

# Function to get the size of the image
def get_size(path):
    img = cv2.imread(path)
    return img.shape[0] if img is not None else 0

# Apply the function to get the size of each image
df['dimension'] = df['filepath'].apply(get_size)
df_filter = df.query('dimension > 60')

# Function to structure the image data
def structuring(path):
    try:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[0]
        if size >= 100:
            gray_resize = cv2.resize(gray, (100,100), cv2.INTER_AREA)
        else:
            gray_resize = cv2.resize(gray, (100,100), cv2.INTER_CUBIC)
        return gray_resize.flatten()
    except:
        return None

# Apply the structuring function to each image path    
df_filter['data'] = df_filter['filepath'].apply(structuring)
# Convert the structured data into a DataFrame
data = pd.DataFrame(df_filter['data'].tolist())
# Rename columns and normalize pixel values
data.columns = [f"pixel_{i}" for i in data.columns]
# Normalize pixel values to [0, 1]
data = data / 255.0
# Add gender and identity columns
data['gender'] = df_filter['gender'].values
data['identity'] = df_filter['identity'].values
# Drop rows with NaN values
data.dropna(inplace=True)
# Save the processed data to a pickle file
dist_identity = df['identity'].value_counts()
# Save the DataFrame to a pickle file
dist_gender = df['gender'].value_counts()

# Save data
pickle.dump(data, open('./data/data_images_100_100.pickle', mode='wb'))



