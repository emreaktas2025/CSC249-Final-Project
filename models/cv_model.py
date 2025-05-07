"""
Traditional CV model for AI vs Real Image Detection
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.features import batch_extract_features, extract_traditional_cv_features


class TraditionalCVModel:
    def __init__(self):
        """
        Initialize the traditional CV model
        """
        self.pipeline = None
        self.feature_importance = None
        self.scaler = None
        self.model = None
    
    def train(self, features_df, optimize=False):
        """
        Train the model on extracted features
        
        Args:
            features_df: DataFrame with extracted features
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            self: Trained model
        """
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                        if col not in ['path', 'label', 'is_real', 'image_path']]
        
        X = features_df[feature_cols]
        y = features_df['is_real']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_SEED
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=config.RANDOM_SEED))
        ])
        
        if optimize:
            # Perform hyperparameter optimization
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
            
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        # Store feature importance
        self.feature_importance = self._get_feature_importance(X.columns)
        
        # Store scaler and model for easier access
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['classifier']
        
        return self
    
    def _get_feature_importance(self, feature_names):
        """
        Get feature importance from the trained model
        
        Args:
            feature_names: Names of the features
            
        Returns:
            pandas.DataFrame: DataFrame with feature importance
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importance
        classifier = self.pipeline.named_steps['classifier']
        importance = classifier.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def predict(self, image_path):
        """
        Predict if an image is real or AI-generated
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Prediction results
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        try:
            # Extract features
            features = extract_traditional_cv_features(image_path)
            
            # Convert to dataframe
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            proba = self.pipeline.predict_proba(feature_df)[0]
            prediction = int(proba[1] > proba[0])
            
            # Return results
            return {
                'is_real_prediction': bool(prediction),
                'real_probability': float(proba[1]),
                'fake_probability': float(proba[0]),
                'confidence': float(max(proba))
            }
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None
    
    def batch_predict(self, df, sample_size=None):
        """
        Make predictions for multiple images
        
        Args:
            df: DataFrame with image paths
            sample_size: Optional number of images to sample
            
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        # Sample the dataframe if requested
        if sample_size is not None and len(df) > sample_size:
            sample_df = df.sample(sample_size, random_state=config.RANDOM_SEED)
        else:
            sample_df = df
        
        # Extract features for all images
        print("Extracting features for batch prediction...")
        features_df = batch_extract_features(sample_df)
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['path', 'label', 'is_real', 'image_path']]
        X = features_df[feature_cols]
        
        # Make predictions
        print("Making predictions...")
        probas = self.pipeline.predict_proba(X)
        
        # Create results dataframe
        results = []
        for i, row in enumerate(sample_df.iterrows()):
            idx, row_data = row
            
            result = {
                'image_path': row_data['path'],
                'actual_label': row_data['label'],
                'actual_is_real': row_data['is_real'],
                'real_probability': float(probas[i][1]),
                'fake_probability': float(probas[i][0]),
                'is_real_prediction': bool(probas[i][1] > probas[i][0]),
                'confidence': float(max(probas[i]))
            }
            
            results.append(result)
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet")
        
        # Save the model
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            self: Loaded model
        """
        self.pipeline = joblib.load(filepath)
        
        # Update scaler and model references
        self.scaler = self.pipeline.named_steps['scaler']
        self.model = self.pipeline.named_steps['classifier']
        
        print(f"Model loaded from {filepath}")
        return self


def train_cv_model(df, sample_size=config.SAMPLE_SIZE['traditional_cv'], optimize=False, save_path=None):
    """
    Train a traditional CV model on the dataset
    
    Args:
        df: DataFrame with image paths
        sample_size: Number of images to use for training
        optimize: Whether to perform hyperparameter optimization
        save_path: Path to save the trained model
        
    Returns:
        TraditionalCVModel: Trained model
    """
    # Extract features
    print(f"Extracting features from {sample_size} images...")
    features_df = batch_extract_features(df, sample_size=sample_size)
    
    # Train model
    print("Training model...")
    model = TraditionalCVModel()
    model.train(features_df, optimize=optimize)
    
    # Save model if requested
    if save_path:
        model.save(save_path)
    
    return model


if __name__ == "__main__":
    # Test traditional CV model
    from utils.dataset import create_dataset
    
    # Create dataset
    df = create_dataset()
    
    # Train model with a small sample
    model = train_cv_model(df, sample_size=50, optimize=False)
    
    # Test on a single image
    test_image = df.iloc[0]['path']
    print(f"Testing on image: {test_image}")
    prediction = model.predict(test_image)
    print(f"Prediction: {prediction}")
    
    # Print top features
    print("\nTop 10 most important features:")
    print(model.feature_importance.head(10))