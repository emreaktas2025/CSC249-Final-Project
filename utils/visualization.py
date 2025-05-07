"""
Visualization utilities for the AI vs Real Image Detection project
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from skimage.filters import gaussian
from skimage.feature import canny
from PIL import Image
import sys
from matplotlib.gridspec import GridSpec

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.features import extract_traditional_cv_features


def visualize_image_with_predictions(image_path, predictions, save_path=None):
    """
    Visualize an image with model predictions
    
    Args:
        image_path: Path to the image
        predictions: Dictionary of model predictions
                    {'model_name': prediction_dict}
        save_path: Optional path to save the figure
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Display image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Image')
    ax1.axis('off')
    
    # Display model predictions
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get actual label from any prediction that has it
    actual_label = None
    for model_name, pred in predictions.items():
        if 'actual_label' in pred:
            actual_label = pred['actual_label']
            break
    
    if actual_label is not None:
        title = f'Model Predictions (Actual: {actual_label.capitalize()})'
    else:
        title = 'Model Predictions'
    
    ax2.set_title(title)
    
    # Create bar chart of predictions
    model_names = list(predictions.keys())
    real_probs = [pred.get('real_probability', 0) for pred in predictions.values()]
    fake_probs = [pred.get('fake_probability', 0) for pred in predictions.values()]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2.bar(x - width/2, real_probs, width, label='Real Probability')
    ax2.bar(x + width/2, fake_probs, width, label='Fake Probability')
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:10] for name in model_names])  # Truncate long names
    ax2.legend()
    
    # Display feature visualizations
    ax3 = fig.add_subplot(gs[1, 0])
    visualize_cv_features(img, ax=ax3)
    
    # Display GPT-4o observations if available
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Check if we have GPT-4o observations
    gpt4o_pred = predictions.get('gpt4o', {})
    if 'key_observations' in gpt4o_pred:
        observations = gpt4o_pred['key_observations']
        reasoning = gpt4o_pred.get('reasoning', '')
        
        obs_text = "GPT-4o Observations:\n"
        for i, obs in enumerate(observations):
            obs_text += f"{i+1}. {obs}\n"
        
        if reasoning:
            obs_text += f"\nReasoning: {reasoning}"
        
        ax4.text(0, 0.5, obs_text, wrap=True, fontsize=10, va='center')
    else:
        ax4.text(0.5, 0.5, "No GPT-4o observations available", 
                 ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_cv_features(image, ax=None):
    """
    Visualize traditional CV features extracted from an image
    
    Args:
        image: Image array (RGB)
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create edge detection
    edges = canny(gray, sigma=config.CANNY_SIGMA)
    
    # Create blurred version
    blurred = gaussian(gray, sigma=config.GAUSSIAN_SIGMA)
    
    # Noise estimation
    noise_est = np.abs(gray.astype(float) - blurred)
    
    # Normalize for visualization
    edges_vis = edges.astype(float)
    noise_vis = noise_est / noise_est.max()
    
    # Create RGB visualization
    vis_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=float)
    
    # Red channel: edges
    vis_img[:,:,0] = edges_vis
    
    # Green channel: original image
    vis_img[:,:,1] = gray.astype(float) / 255
    
    # Blue channel: noise estimation
    vis_img[:,:,2] = noise_vis
    
    # Display
    ax.imshow(vis_img)
    ax.set_title('CV Features: Red=Edges, Green=Original, Blue=Noise')
    ax.axis('off')
    
    return ax


def visualize_model_errors(error_analysis, df, save_dir=None):
    """
    Visualize examples of model errors
    
    Args:
        error_analysis: Error analysis from analyze_errors
        df: DataFrame with dataset information
        save_dir: Optional directory to save the figures
    """
    # Check if we have error analysis
    if error_analysis is None or 'fp_images' not in error_analysis:
        print("No error analysis available")
        return
    
    model_name = error_analysis['model_name']
    
    # Create save directory if needed
    if save_dir is not None:
        error_dir = os.path.join(save_dir, f"{model_name}_errors")
        os.makedirs(error_dir, exist_ok=True)
    else:
        error_dir = None
    
    # Sample false positives (fake images classified as real)
    fp_images = error_analysis['fp_images']
    if fp_images:
        print(f"\nFalse positives for {model_name} (up to 4):")
        for i, img_path in enumerate(fp_images[:4]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"False Positive: Fake image classified as Real")
            plt.axis('off')
            
            if error_dir:
                save_path = os.path.join(error_dir, f"fp_{i}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    # Sample false negatives (real images classified as fake)
    fn_images = error_analysis['fn_images']
    if fn_images:
        print(f"\nFalse negatives for {model_name} (up to 4):")
        for i, img_path in enumerate(fn_images[:4]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"False Negative: Real image classified as Fake")
            plt.axis('off')
            
            if error_dir:
                save_path = os.path.join(error_dir, f"fn_{i}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()


def visualize_sample_images(df, num_samples=5, save_dir=None):
    """
    Visualize sample images from the dataset
    
    Args:
        df: DataFrame with dataset information
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save the figures
    """
    # Sample real images
    real_df = df[df['is_real'] == 1].sample(num_samples)
    fake_df = df[df['is_real'] == 0].sample(num_samples)
    
    # Create save directory if needed
    if save_dir is not None:
        samples_dir = os.path.join(save_dir, "sample_images")
        os.makedirs(samples_dir, exist_ok=True)
    else:
        samples_dir = None
    
    # Display real images
    print("\nSample real images:")
    for i, (_, row) in enumerate(real_df.iterrows()):
        img_path = row['path']
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Real Image")
        plt.axis('off')
        
        if samples_dir:
            save_path = os.path.join(samples_dir, f"real_{i}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Display fake images
    print("\nSample AI-generated images:")
    for i, (_, row) in enumerate(fake_df.iterrows()):
        img_path = row['path']
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"AI-Generated Image")
        plt.axis('off')
        
        if samples_dir:
            save_path = os.path.join(samples_dir, f"fake_{i}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_feature_importance(importance_df, top_n=20, save_path=None):
    """
    Visualize feature importance from the traditional CV model
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to display
        save_path: Optional path to save the figure
    """
    # Select top features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Most Important CV Features')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {save_path}")
    
    plt.show()


def visualize_image_with_features(image_path, save_path=None):
    """
    Visualize an image with its extracted features
    
    Args:
        image_path: Path to the image
        save_path: Optional path to save the figure
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract features
    features = extract_traditional_cv_features(image_path)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Gaussian blur
    blurred = gaussian(gray, sigma=config.GAUSSIAN_SIGMA)
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur')
    axes[0, 2].axis('off')
    
    # Edge detection
    edges = canny(gray, sigma=config.CANNY_SIGMA)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title(f'Edge Detection (Density: {features["edge_density"]:.4f})')
    axes[1, 0].axis('off')
    
    # SIFT features
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    sift_img = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    axes[1, 1].imshow(sift_img)
    axes[1, 1].set_title(f'SIFT Features: {len(keypoints)} keypoints')
    axes[1, 1].axis('off')
    
    # Noise estimation
    noise_est = np.abs(gray.astype(float) - blurred)
    noise_vis = noise_est / noise_est.max()
    axes[1, 2].imshow(noise_vis, cmap='hot')
    axes[1, 2].set_title(f'Noise Estimation (Mean: {features["noise_mean"]:.4f})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature visualization to {save_path}")
    
    plt.show()
    
    # Print key feature values
    print("Key Feature Values:")
    important_features = [
        'edge_density', 'sift_density', 'noise_mean', 'gaussian_diff_mean',
        'laplacian_var', 'mean_red', 'mean_green', 'mean_blue'
    ]
    
    for feat in important_features:
        if feat in features:
            print(f"{feat}: {features[feat]:.6f}")
            
def visualize_model_predictions(image_path, predictions, save_path=None):
    """
    Alias for visualize_image_with_predictions for backward compatibility
    """
    return visualize_image_with_predictions(image_path, predictions, save_path)


if __name__ == "__main__":
    # Test visualization functions
    from utils.dataset import create_dataset
    
    # Create dataset
    df = create_dataset()
    
    # Test image with features
    test_image = df.iloc[0]['path']
    visualize_image_with_features(test_image)
    
    # Test sample images
    visualize_sample_images(df, num_samples=2)
    
    # Test model predictions visualization
    test_predictions = {
        'clip': {
            'is_real_prediction': True,
            'real_probability': 0.8,
            'fake_probability': 0.2,
            'confidence': 0.8,
            'actual_label': 'real'
        },
        'cv': {
            'is_real_prediction': True,
            'real_probability': 0.7,
            'fake_probability': 0.3,
            'confidence': 0.7
        },
        'gpt4o': {
            'is_real_prediction': True,
            'real_probability': 0.9,
            'fake_probability': 0.1,
            'confidence': 0.9,
            'key_observations': [
                'Natural lighting and shadows',
                'Realistic texture details',
                'Complex background elements match scene',
                'Natural facial features and expressions'
            ],
            'reasoning': 'The image shows consistent lighting, natural textures, and realistic details that would be difficult for AI to generate perfectly.'
        }
    }
    
    visualize_image_with_predictions(test_image, test_predictions)