"""
Dataset handling functions for the AI vs Real Image Detection project
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_dataset(real_folder=config.REAL_IMAGES_DIR, 
                  fake_folder=config.FAKE_IMAGES_DIR, 
                  csv_path=os.path.join(config.DATA_DIR, 'dataset.csv')):
    """
    Create a CSV dataset from image folders with labels and paths
    
    Args:
        real_folder: Directory containing real images
        fake_folder: Directory containing AI-generated (fake) images
        csv_path: Path to save the dataset CSV
        
    Returns:
        pandas.DataFrame: The dataset with image paths and labels
    """
    # Get lists of image files
    real_images = glob.glob(os.path.join(real_folder, '*.*'))
    fake_images = glob.glob(os.path.join(fake_folder, '*.*'))
    
    # Check if there are images
    if len(real_images) == 0:
        print(f"No real images found in {real_folder}")
    if len(fake_images) == 0:
        print(f"No fake images found in {fake_folder}")
    
    # Create dataframe
    data = []
    for img_path in real_images:
        data.append({'path': img_path, 'label': 'real', 'is_real': 1})
    
    for img_path in fake_images:
        data.append({'path': img_path, 'label': 'fake', 'is_real': 0})
    
    # Shuffle the data
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Created dataset with {len(df)} images: {df['label'].value_counts().to_dict()}")
    
    return df


def split_dataset(df, test_size=config.TEST_SPLIT, random_state=config.RANDOM_SEED):
    """
    Split dataset into training and testing sets
    
    Args:
        df: DataFrame with the dataset
        test_size: Proportion of the dataset to include in the test split
        random_state: Seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label']  # Ensure balanced classes in both splits
    )
    
    print(f"Train set: {len(train_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    return train_df, test_df


def sample_dataset(df, sample_size, random_state=config.RANDOM_SEED):
    """
    Create a balanced sample from the dataset
    
    Args:
        df: DataFrame with the dataset
        sample_size: Number of samples to include
        random_state: Seed for reproducibility
        
    Returns:
        pandas.DataFrame: Sample dataset
    """
    if len(df) <= sample_size:
        return df
    
    # Get balanced sample with equal number of real and fake images
    real_df = df[df['is_real'] == 1]
    fake_df = df[df['is_real'] == 0]
    
    # Determine sample size per class
    samples_per_class = min(sample_size // 2, min(len(real_df), len(fake_df)))
    
    # Sample from each class
    real_sample = real_df.sample(samples_per_class, random_state=random_state)
    fake_sample = fake_df.sample(samples_per_class, random_state=random_state)
    
    # Combine and shuffle
    sample_df = pd.concat([real_sample, fake_sample])
    sample_df = sample_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Created balanced sample with {len(sample_df)} images")
    
    return sample_df


def check_image_validity(df):
    """
    Check if all images in the dataset can be opened
    
    Args:
        df: DataFrame with image paths
        
    Returns:
        pandas.DataFrame: Dataset with only valid images
    """
    valid_paths = []
    invalid_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        try:
            img = cv2.imread(row['path'])
            if img is None:
                invalid_paths.append(row['path'])
            else:
                valid_paths.append(row)
        except Exception as e:
            print(f"Error with image {row['path']}: {e}")
            invalid_paths.append(row['path'])
    
    if invalid_paths:
        print(f"Found {len(invalid_paths)} invalid images")
        for path in invalid_paths:
            print(f" - {path}")
    
    valid_df = pd.DataFrame(valid_paths)
    
    return valid_df


if __name__ == "__main__":
    # Test the dataset creation
    df = create_dataset()
    print(df.head())
    
    # Split dataset
    train_df, test_df = split_dataset(df)
    
    # Sample dataset
    sample_df = sample_dataset(train_df, sample_size=10)
    print(sample_df)