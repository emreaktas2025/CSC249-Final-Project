"""
Feature extraction utilities for the AI vs Real Image Detection project
"""

import os
import numpy as np
import cv2
from skimage.filters import gaussian
from skimage.feature import canny
from tqdm import tqdm
import pandas as pd
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_traditional_cv_features(image_path):
    """
    Extract traditional computer vision features from an image
    
    Args:
        image_path: Path to the image
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    features = {}
    
    # Basic image properties
    features['height'] = img.shape[0]
    features['width'] = img.shape[1]
    features['aspect_ratio'] = img.shape[1] / img.shape[0]
    
    # Color analysis
    features['mean_red'] = np.mean(img[:,:,0])
    features['mean_green'] = np.mean(img[:,:,1])
    features['mean_blue'] = np.mean(img[:,:,2])
    features['std_red'] = np.std(img[:,:,0])
    features['std_green'] = np.std(img[:,:,1])
    features['std_blue'] = np.std(img[:,:,2])
    
    # Color histogram features
    for i, color in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        for bin_idx, value in enumerate(hist):
            features[f'{color}_hist_bin_{bin_idx}'] = value
    
    # Gaussian blur - measure difference between original and blurred
    blurred = gaussian(gray, sigma=config.GAUSSIAN_SIGMA)
    features['gaussian_diff_mean'] = np.mean(np.abs(gray - blurred))
    features['gaussian_diff_std'] = np.std(np.abs(gray - blurred))
    
    # Edge detection
    edges = canny(gray, sigma=config.CANNY_SIGMA)
    features['edge_density'] = np.mean(edges)
    
    # SIFT feature count (approximated density)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    if keypoints is not None:
        features['sift_density'] = len(keypoints) / (img.shape[0] * img.shape[1])
        features['sift_count'] = len(keypoints)
    else:
        features['sift_density'] = 0
        features['sift_count'] = 0
    
    # Noise estimation
    # Use the difference between the image and its blurred version as a proxy for noise
    noise_est = np.abs(gray.astype(float) - blurred)
    features['noise_mean'] = np.mean(noise_est)
    features['noise_std'] = np.std(noise_est)
    
    # Laplacian variance (measure of image blurriness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var'] = laplacian.var()
    
    # Local Binary Pattern (LBP) for texture analysis
    radius = 3
    n_points = 8 * radius
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float") / (hist.sum() + 1e-7)
    for i, value in enumerate(hist):
        features[f'lbp_hist_bin_{i}'] = value
    
    return features


def batch_extract_features(df, sample_size=None):
    """
    Extract traditional CV features for multiple images
    
    Args:
        df: DataFrame with image paths
        sample_size: Optional number of images to sample
        
    Returns:
        pandas.DataFrame: DataFrame with extracted features
    """
    # Sample the dataframe if requested
    if sample_size is not None and len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=config.RANDOM_SEED)
    else:
        sample_df = df
    
    # Extract features for each image
    all_features = []
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Extracting CV features"):
        try:
            features = extract_traditional_cv_features(row['path'])
            features['path'] = row['path']
            features['label'] = row['label']
            features['is_real'] = row['is_real']
            all_features.append(features)
        except Exception as e:
            print(f"Error extracting features for {row['path']}: {e}")
    
    # Convert to dataframe
    features_df = pd.DataFrame(all_features)
    
    return features_df


if __name__ == "__main__":
    # Test feature extraction on a sample image
    import matplotlib.pyplot as plt
    from utils.dataset import create_dataset
    
    df = create_dataset()
    test_image = df.iloc[0]['path']
    
    features = extract_traditional_cv_features(test_image)
    print(f"Extracted {len(features)} features")
    for key, value in list(features.items())[:10]:  # Show first 10 features
        print(f"{key}: {value}")
    
    # Test batch feature extraction
    features_df = batch_extract_features(df, sample_size=5)
    print(features_df.shape)