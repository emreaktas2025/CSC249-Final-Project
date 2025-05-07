"""
CLIP model implementation for AI vs Real Image Detection
"""

import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CLIPClassifier:
    def __init__(self, model_name=config.CLIP_MODEL_NAME):
        """
        Initialize the CLIP classifier
        
        Args:
            model_name: Name of the CLIP model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"Loaded CLIP model: {model_name}")
        
        # Generate text prompts
        self.generate_text_prompts()
    
    def generate_text_prompts(self):
        """
        Generate text prompts for classifying real vs AI-generated images
        """
        # Detailed prompts
        self.detailed_prompts = [
            "a real photograph taken with a camera",
            "a genuine photograph of reality",
            "an authentic photograph with natural lighting and details",
            "a natural photograph with realistic imperfections",
            "a photo showing real-world content with realistic textures",
            "an AI-generated image created by a computer",
            "a fake image created by generative AI",
            "an artificial image with typical AI artifacts",
            "a synthetic image with unnatural patterns and details",
            "a computer-generated image with inconsistent lighting"
        ]
        
        # Simple prompts for binary classification
        self.binary_prompts = [
            "a real photograph",
            "an AI-generated image"
        ]
        
        # Encode the text prompts
        self.encode_text_prompts()
    
    def encode_text_prompts(self):
        """
        Encode the text prompts using CLIP
        """
        # Encode detailed prompts
        tokens = clip.tokenize(self.detailed_prompts).to(self.device)
        with torch.no_grad():
            self.detailed_features = self.model.encode_text(tokens)
            self.detailed_features = self.detailed_features / self.detailed_features.norm(dim=-1, keepdim=True)
        
        # Encode binary prompts
        tokens = clip.tokenize(self.binary_prompts).to(self.device)
        with torch.no_grad():
            self.binary_features = self.model.encode_text(tokens)
            self.binary_features = self.binary_features / self.binary_features.norm(dim=-1, keepdim=True)
    
    def predict_image(self, image_path, detailed=False):
        """
        Predict if an image is real or AI-generated
        
        Args:
            image_path: Path to the image
            detailed: Whether to use detailed prompts
            
        Returns:
            dict: Prediction results
        """
        # Open and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity with text prompts
        if detailed:
            text_features = self.detailed_features
            prompts = self.detailed_prompts
        else:
            text_features = self.binary_features
            prompts = self.binary_prompts
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        similarity = similarity.cpu().numpy()[0]
        
        # Organize results
        similarities = [(prompts[i], float(similarity[i])) for i in range(len(prompts))]
        
        if detailed:
            # Calculate aggregate scores for real vs. fake categories
            real_prompts = [p for p in prompts if "real" in p or "genuine" in p or "authentic" in p or "natural" in p]
            fake_prompts = [p for p in prompts if "AI" in p or "fake" in p or "artificial" in p or "synthetic" in p or "computer" in p]
            
            real_indices = [i for i, p in enumerate(prompts) if p in real_prompts]
            fake_indices = [i for i, p in enumerate(prompts) if p in fake_prompts]
            
            real_score = sum(similarity[i] for i in real_indices)
            fake_score = sum(similarity[i] for i in fake_indices)
            
            # Normalize
            total = real_score + fake_score
            real_prob = real_score / total if total > 0 else 0.5
            fake_prob = fake_score / total if total > 0 else 0.5
        else:
            # Binary classification
            real_prob = similarity[0]
            fake_prob = similarity[1]
        
        is_real_prediction = real_prob > fake_prob
        confidence = max(real_prob, fake_prob)
        
        return {
            'all_similarities': similarities,
            'real_score': float(real_prob),
            'fake_score': float(fake_prob),
            'is_real_prediction': bool(is_real_prediction),
            'confidence': float(confidence)
        }
    
    def batch_predict(self, df, sample_size=None, detailed=False):
        """
        Make predictions for multiple images
        
        Args:
            df: DataFrame with image paths
            sample_size: Optional number of images to sample
            detailed: Whether to use detailed prompts
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        # Sample the dataframe if requested
        if sample_size is not None and len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=config.RANDOM_SEED)
        else:
            sample_df = df
        
        # Process each image
        results = []
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing with CLIP"):
            pred = self.predict_image(row['path'], detailed=detailed)
            if pred is not None:
                # Add metadata
                pred['image_path'] = row['path']
                pred['actual_label'] = row['label']
                pred['actual_is_real'] = row['is_real']
                
                results.append(pred)
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        return results_df


if __name__ == "__main__":
    # Test CLIP classifier
    from utils.dataset import create_dataset
    
    # Create dataset
    df = create_dataset()
    
    # Initialize CLIP classifier
    clip_classifier = CLIPClassifier()
    
    # Test on a single image
    test_image = df.iloc[0]['path']
    print(f"Testing on image: {test_image}")
    prediction = clip_classifier.predict_image(test_image, detailed=True)
    print(f"Prediction: {prediction}")
    
    # Test batch prediction
    results = clip_classifier.batch_predict(df, sample_size=5, detailed=True)
    print(results)