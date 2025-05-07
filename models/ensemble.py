"""
Ensemble model combining multiple approaches for AI vs Real Image Detection
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EnsembleModel:
    def __init__(self, models=None):
        """
        Initialize the ensemble model
        
        Args:
            models: Dictionary of models to use in the ensemble
                   {'name': model_instance}
        """
        self.models = models if models is not None else {}
        self.weights = None
    
    def add_model(self, name, model):
        """
        Add a model to the ensemble
        
        Args:
            name: Name of the model
            model: Model instance
        """
        self.models[name] = model
    
    def set_weights(self, weights):
        """
        Set weights for the ensemble models
        
        Args:
            weights: Dictionary of weights for each model
                    {'name': weight}
        """
        self.weights = weights
    
    def predict(self, image_path):
        """
        Make an ensemble prediction for an image
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Ensemble prediction
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            try:
                pred = model.predict(image_path)
                if pred is not None:
                    predictions[name] = pred
            except Exception as e:
                print(f"Error with {name} model: {e}")
        
        if not predictions:
            print(f"No valid predictions for {image_path}")
            return None
        
        # Combine predictions
        return self._combine_predictions(predictions)
    
    def _combine_predictions(self, predictions):
        """
        Combine predictions from multiple models
        
        Args:
            predictions: Dictionary of predictions from each model
            
        Returns:
            dict: Combined prediction
        """
        # Extract probabilities
        real_probs = []
        fake_probs = []
        confidences = []
        
        for name, pred in predictions.items():
            # Skip None predictions
            if pred is None or 'real_probability' not in pred:
                continue
                
            # Get weight for this model
            weight = 1.0
            if self.weights and name in self.weights:
                weight = self.weights[name]
            
            real_probs.append(pred['real_probability'] * weight)
            fake_probs.append(pred['fake_probability'] * weight)
            confidences.append(pred['confidence'] * weight)
        
        # Calculate weighted averages
        if not real_probs:
            return None
            
        total_weight = sum(self.weights.values()) if self.weights else len(real_probs)
        
        avg_real_prob = sum(real_probs) / total_weight
        avg_fake_prob = sum(fake_probs) / total_weight
        avg_confidence = sum(confidences) / total_weight
        
        # Make ensemble prediction
        is_real_pred = avg_real_prob > avg_fake_prob
        
        # Return results
        return {
            'is_real_prediction': bool(is_real_pred),
            'real_probability': float(avg_real_prob),
            'fake_probability': float(avg_fake_prob),
            'confidence': float(avg_confidence),
            'model_predictions': predictions
        }
    
    def batch_predict(self, df, sample_size=None):
        """
        Make ensemble predictions for multiple images
        
        Args:
            df: DataFrame with image paths
            sample_size: Optional number of images to sample
            
        Returns:
            pandas.DataFrame: DataFrame with ensemble predictions
        """
        # Sample the dataframe if requested
        if sample_size is not None and len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=config.RANDOM_SEED)
        else:
            sample_df = df
        
        # Get individual model predictions
        model_results = {}
        for name, model in self.models.items():
            try:
                print(f"Getting predictions from {name} model...")
                model_results[name] = model.batch_predict(sample_df)
            except Exception as e:
                print(f"Error with {name} model: {e}")
        
        # Combine predictions
        ensemble_results = []
        
        for idx, row in sample_df.iterrows():
            image_path = row['path']
            
            # Get predictions for this image from each model
            image_predictions = {}
            for name, results_df in model_results.items():
                # Find row for this image
                model_row = results_df[results_df['image_path'] == image_path]
                if not model_row.empty:
                    # Convert row to dict
                    image_predictions[name] = model_row.iloc[0].to_dict()
            
            # Combine predictions
            if image_predictions:
                ensemble_pred = self._combine_predictions(image_predictions)
                
                if ensemble_pred:
                    # Add metadata
                    ensemble_pred['image_path'] = image_path
                    ensemble_pred['actual_label'] = row['label']
                    ensemble_pred['actual_is_real'] = row['is_real']
                    
                    ensemble_results.append(ensemble_pred)
        
        # Convert to dataframe
        results_df = pd.DataFrame(ensemble_results)
        
        return results_df
    
    def evaluate(self, results_df):
        """
        Evaluate ensemble performance
        
        Args:
            results_df: DataFrame with ensemble predictions
            
        Returns:
            dict: Evaluation metrics
        """
        # Calculate accuracy
        y_true = results_df['actual_is_real']
        y_pred = results_df['is_real_prediction']
        
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract model-specific metrics if available
        model_metrics = {}
        for name in self.models.keys():
            model_col = f'{name}_prediction'
            if model_col in results_df.columns:
                model_acc = accuracy_score(y_true, results_df[model_col])
                model_metrics[name] = {'accuracy': model_acc}
        
        # Return all metrics
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'model_metrics': model_metrics
        }


def create_default_ensemble(clip_model=None, cv_model=None, gpt4o_model=None):
    """
    Create an ensemble with default models and weights
    
    Args:
        clip_model: CLIP model instance
        cv_model: Traditional CV model instance
        gpt4o_model: GPT-4o model instance
        
    Returns:
        EnsembleModel: Ensemble model
    """
    ensemble = EnsembleModel()
    
    # Add available models
    if clip_model:
        ensemble.add_model('clip', clip_model)
    
    if cv_model:
        ensemble.add_model('cv', cv_model)
    
    if gpt4o_model:
        ensemble.add_model('gpt4o', gpt4o_model)
    
    # Set default weights (GPT-4o has highest weight, CLIP second, CV lowest)
    weights = {
        'clip': 0.35,
        'cv': 0.15,
        'gpt4o': 0.5
    }
    
    # Filter weights to only include available models
    available_weights = {name: weight for name, weight in weights.items() 
                        if name in ensemble.models}
    
    # Normalize weights
    if available_weights:
        total = sum(available_weights.values())
        normalized_weights = {name: weight/total for name, weight in available_weights.items()}
        ensemble.set_weights(normalized_weights)
    
    return ensemble


if __name__ == "__main__":
    # Test ensemble model
    from utils.dataset import create_dataset
    from models.clip_model import CLIPClassifier
    from models.cv_model import TraditionalCVModel, train_cv_model
    
    # Create dataset
    df = create_dataset()
    
    # Create models
    clip_model = CLIPClassifier()
    cv_model = train_cv_model(df, sample_size=50)
    
    # Create ensemble
    ensemble = create_default_ensemble(clip_model, cv_model)
    
    # Test on a single image
    test_image = df.iloc[0]['path']
    print(f"Testing on image: {test_image}")
    prediction = ensemble.predict(test_image)
    print(f"Ensemble prediction: {prediction}")
    
    # Test batch prediction
    results = ensemble.batch_predict(df, sample_size=5)
    print(results)