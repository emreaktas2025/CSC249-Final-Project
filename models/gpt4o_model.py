"""
GPT-4o model integration for AI vs Real Image Detection
"""

import os
import base64
import json
import requests
import time
from tqdm import tqdm
import pandas as pd
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GPT4oClassifier:
    def __init__(self, api_key=None):
        """
        Initialize the GPT-4o classifier
        
        Args:
            api_key: OpenAI API key (if None, will try to use the one in config)
        """
        # Set API key
        self.api_key = api_key if api_key is not None else config.GPT4O_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it in config.py or pass it to the constructor.")
        
        # API settings
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prompt settings
        self.system_prompt = """
        You are an expert in detecting AI-generated images versus real photographs.
        Analyze the provided image and determine whether it is a real photograph or an AI-generated image.
        Focus on these aspects:
        1. Consistency of lighting and shadows
        2. Texture details (especially in skin, hair, fabrics)
        3. Background coherence and realism
        4. Any artifacts or unnatural elements
        5. Consistency of perspective and scale
        6. Realistic imperfections and noise
        
        Return your analysis in JSON format with these fields:
        {
            "is_real": boolean (true if real photo, false if AI-generated),
            "confidence": float (between 0 and 1),
            "key_observations": [list of 3-5 strings - key elements that influenced your decision],
            "reasoning": string (brief explanation of your decision)
        }
        """
    
    def encode_image(self, image_path):
        """
        Encode image to base64
        
        Args:
            image_path: Path to the image
            
        Returns:
            str: Base64-encoded image
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def classify_image(self, image_path, max_retries=3, retry_delay=2):
        """
        Classify an image using GPT-4o
        
        Args:
            image_path: Path to the image
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            dict: Classification results
        """
        # Encode image
        base64_image = self.encode_image(image_path)
        if base64_image is None:
            return {
                "is_real": None,
                "confidence": 0,
                "key_observations": ["Failed to encode image"],
                "reasoning": f"Error: Could not encode image {image_path}",
                "processing_time": 0
            }
        
        # Prepare the API request
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this image and determine if it's real or AI-generated."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        
        # Track start time
        start_time = time.time()
        
        # Make the API request with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Parse the response
                result = response.json()
                content = json.loads(result['choices'][0]['message']['content'])
                
                # Add processing time
                content['processing_time'] = processing_time
                
                return content
                
            except Exception as e:
                print(f"Error classifying image (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # Return error result on final failure
                    return {
                        "is_real": None,
                        "confidence": 0,
                        "key_observations": ["API error after multiple retries"],
                        "reasoning": f"Error: {str(e)}",
                        "processing_time": time.time() - start_time
                    }
    
    def batch_classify(self, df, sample_size=None):
        """
        Classify multiple images using GPT-4o
        
        Args:
            df: DataFrame with image paths
            sample_size: Optional number of images to sample
            
        Returns:
            pandas.DataFrame: DataFrame with classifications
        """
        # Sample the dataframe if requested
        if sample_size is not None and len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=config.RANDOM_SEED)
        else:
            sample_df = df
        
        # Process each image
        results = []
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing with GPT-4o"):
            # Get prediction
            pred = self.classify_image(row['path'])
            
            # Add metadata
            pred['image_path'] = row['path']
            pred['actual_label'] = row['label']
            pred['actual_is_real'] = row['is_real']
            
            # Format for consistency with other models
            if pred['is_real'] is not None:
                pred['is_real_prediction'] = pred['is_real']
                
                # Calculate probabilities based on confidence
                if pred['is_real']:
                    pred['real_probability'] = pred['confidence']
                    pred['fake_probability'] = 1 - pred['confidence']
                else:
                    pred['fake_probability'] = pred['confidence']
                    pred['real_probability'] = 1 - pred['confidence']
            else:
                # Handle error case
                pred['is_real_prediction'] = None
                pred['real_probability'] = 0
                pred['fake_probability'] = 0
            
            results.append(pred)
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def batch_predict(self, df, sample_size=None):
        """
        Alias for batch_classify for consistency with other model interfaces
        
        Args:
            df: DataFrame with image paths
            sample_size: Optional number of images to sample
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        return self.batch_classify(df, sample_size)


if __name__ == "__main__":
    # Test GPT-4o classifier
    from utils.dataset import create_dataset
    
    # Create dataset
    df = create_dataset()
    
    # Check if API key is available
    if not config.GPT4O_API_KEY:
        print("Warning: No OpenAI API key found in config. Set GPT4O_API_KEY in config.py to test this module.")
    else:
        try:
            # Initialize GPT-4o classifier
            gpt4o_classifier = GPT4oClassifier()
            
            # Test on a single image
            test_image = df.iloc[0]['path']
            print(f"Testing on image: {test_image}")
            prediction = gpt4o_classifier.classify_image(test_image)
            print(f"Prediction: {prediction}")
            
            # Test batch classification with a very small sample
            results = gpt4o_classifier.batch_classify(df, sample_size=2)
            print(results)
        except Exception as e:
            print(f"Error testing GPT-4o classifier: {e}")