from pathlib import Path
import cv2
import pandas as pd
import pickle

# Get all female and male cropped image paths
DATA_DIR = Path('./data/images/crop')
OUTPUT_PATH = Path('./data/data_images_100_100.pickle')


def collect_paths() -> pd.DataFrame:
    # Collect paths for female images
    female_paths = list(DATA_DIR.joinpath('female').rglob('*.jpg')) + list(
        DATA_DIR.joinpath('female').rglob('*.jpeg')) + list(
            DATA_DIR.joinpath('female').rglob('*.png'))
    # Collect paths for male images
    male_paths = list(DATA_DIR.joinpath('male').rglob('*.jpg')) + list(
        DATA_DIR.joinpath('male').rglob('*.jpeg')) + list(
        DATA_DIR.joinpath('male').rglob('*.png'))
    
    # Create DataFrames for each gender
    df_female = pd.DataFrame(female_paths, columns=['filepath'])
    df_female['gender'] = 'female'
    df_female['identity'] = df_female['filepath'].apply(extract_identity)
    df_male = pd.DataFrame(male_paths, columns=['filepath'])
    df_male['gender'] = 'male'
    df_male['identity'] = df_male['filepath'].apply(extract_identity)
    return pd.concat((df_female, df_male), axis=0, ignore_index=True)

def extract_identity(path: Path) -> str:
    # Extract identity from the file path
    # The identity is assumed to be the folder name after 'known' or 'unknown'
    parts = str(path).replace('\\', '/').split('/')
    if 'known' in parts:
        return parts[parts.index('known') + 1]
    elif 'unknown' in parts:
        return 'unknown'
    return 'unknown'  # fallback

def get_size(path: Path) -> int:
    # Get the height of the image at the given path
    # Returns 0 if the image cannot be read
    img = cv2.imread(str(path))
    return img.shape[0] if img is not None else 0

# Function to structure the image data
def structuring(path: Path):
    try:
        img = cv2.imread(str(path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[0]
        if size >= 100:
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_AREA)
        else:
            gray_resize = cv2.resize(gray, (100, 100), cv2.INTER_CUBIC)
        return gray_resize.flatten()
    except Exception:
        return None

def main():
    df = collect_paths() # Collect paths
    # Filter out images that are too small
    df['dimensions'] = df['filepath'].apply(get_size)
    df_filter = df.query('dimensions > 60')
    df_filter['data'] = df_filter['filepath'].apply(structuring) # Structure the data
    data = pd.DataFrame(df_filter['data'].tolist()) # Convert to DataFrame
    data.columns = [f"pixel_{i}" for i in data.columns] # Rename columns to pixel_0, pixel_1, etc.
    data = data / 255.0  # Normalize pixel values
    data['gender'] = df_filter['gender'].values # Add gender column
    data['identity'] = df_filter['identity'].values # Add identity column
    data.dropna(inplace=True)  # Drop rows with None values

    print(f"Total images after filtering: {len(df_filter)}") 
    print(f"Total images after structuring: {len(data)}")
    with open(OUTPUT_PATH, 'wb') as f: # Save the data to a pickle file
        pickle.dump(data, f)

if __name__ == "__main__": # Run the main function
    main()
    print(f"Data saved to {OUTPUT_PATH}")



