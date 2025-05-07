"""
Configuration settings for the AI vs Real Image Detection project
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
REAL_IMAGES_DIR = os.path.join(DATA_DIR, 'real')
FAKE_IMAGES_DIR = os.path.join(DATA_DIR, 'fake')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create necessary directories if they don't exist
os.makedirs(REAL_IMAGES_DIR, exist_ok=True)
os.makedirs(FAKE_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model settings
CLIP_MODEL_NAME = "ViT-B/32"

# Get API key from environment variable
GPT4O_API_KEY = os.getenv('GPT4O_API_KEY')
if not GPT4O_API_KEY:
    raise ValueError("Please set the GPT4O_API_KEY environment variable")

# Dataset settings
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Feature extraction settings
GAUSSIAN_SIGMA = 2
CANNY_SIGMA = 1

# Experiment settings
SAMPLE_SIZE = {
    'traditional_cv': 100,
    'clip': 100,
    'gpt4o': 20  # Smaller due to API costs
}

# Visualization settings
FIGURE_SIZE = (12, 8)
FONTSIZE = 12