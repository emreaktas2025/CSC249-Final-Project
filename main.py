#!/usr/bin/env python3
"""
AI vs Real Image Detection Project - Main Script

This script provides a command-line interface to run the project components:
- Dataset processing
- Traditional CV model training and evaluation
- CLIP model evaluation
- GPT-4o model evaluation (if API key available)
- Ensemble model creation and evaluation
- Visualization and analysis
"""

import os
import argparse
import pandas as pd
import time
import sys
import numpy as np
from pathlib import Path

# Import project modules
import config
from utils.dataset import create_dataset, split_dataset, sample_dataset, check_image_validity
from utils.features import batch_extract_features
from utils.evaluation import (
    evaluate_model, print_evaluation_report, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, compare_models,
    plot_model_comparison, analyze_errors, print_error_analysis
)
from utils.visualization import (
    visualize_sample_images, visualize_image_with_features,
    visualize_image_with_predictions, visualize_feature_importance, 
    visualize_model_errors
)
from models.cv_model import TraditionalCVModel, train_cv_model
from models.clip_model import CLIPClassifier
from models.ensemble import EnsembleModel, create_default_ensemble


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI vs Real Image Detection')
    
    # Main commands
    parser.add_argument('--prepare-data', action='store_true',
                        help='Prepare the dataset')
    parser.add_argument('--train-cv-model', action='store_true',
                        help='Train the traditional CV model')
    parser.add_argument('--evaluate-models', action='store_true',
                        help='Evaluate all available models')
    parser.add_argument('--analyze-image', type=str,
                        help='Analyze a specific image')
    parser.add_argument('--visualize-samples', action='store_true',
                        help='Visualize sample images from the dataset')
    
    # Dataset options
    parser.add_argument('--real-dir', type=str, default=config.REAL_IMAGES_DIR,
                        help='Directory with real images')
    parser.add_argument('--fake-dir', type=str, default=config.FAKE_IMAGES_DIR,
                        help='Directory with fake images')
    parser.add_argument('--sample-size', type=int, default=100,
                        help='Number of images to sample for analysis')
    
    # Model options
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--gpt4o-key', type=str, default=config.GPT4O_API_KEY,
                        help='OpenAI API key for GPT-4o')
    
    # Output options
    parser.add_argument('--save-results', action='store_true',
                        help='Save results and visualizations')
    parser.add_argument('--output-dir', type=str, default=config.RESULTS_DIR,
                        help='Directory to save results')
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare the dataset"""
    print("=== Preparing Dataset ===")
    
    # Create dataset CSV
    df = create_dataset(args.real_dir, args.fake_dir)
    
    # Check image validity
    print("Checking image validity...")
    df = check_image_validity(df)
    
    # Split dataset
    train_df, test_df = split_dataset(df)
    
    # Save splits if requested
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
        print(f"Saved dataset splits to {args.output_dir}")
    
    return train_df, test_df


def train_models(args, train_df):
    """Train models"""
    models = {}
    
    # Train traditional CV model
    if args.train_cv_model:
        print("\n=== Training Traditional CV Model ===")
        
        cv_model = train_cv_model(
            train_df, 
            sample_size=config.SAMPLE_SIZE['traditional_cv'],
            optimize=args.optimize,
            save_path=os.path.join(args.output_dir, 'cv_model.joblib') if args.save_results else None
        )
        
        models['cv'] = cv_model
        
        # Visualize feature importance if requested
        if args.save_results and cv_model.feature_importance is not None:
            save_path = os.path.join(args.output_dir, 'feature_importance.png')
            visualize_feature_importance(cv_model.feature_importance, save_path=save_path)
    
    # Load CLIP model
    print("\n=== Loading CLIP Model ===")
    clip_model = CLIPClassifier()
    models['clip'] = clip_model
    
    # Load GPT-4o model if API key available
    if args.gpt4o_key:
        print("\n=== Initializing GPT-4o Model ===")
        try:
            from models.gpt4o_model import GPT4oClassifier
            gpt4o_model = GPT4oClassifier(api_key=args.gpt4o_key)
            models['gpt4o'] = gpt4o_model
            print("GPT-4o model initialized successfully")
        except Exception as e:
            print(f"Error initializing GPT-4o model: {e}")
    else:
        print("No OpenAI API key provided. Skipping GPT-4o model.")
    
    # Create ensemble model
    print("\n=== Creating Ensemble Model ===")
    ensemble = create_default_ensemble(
        clip_model=models.get('clip'),
        cv_model=models.get('cv'),
        gpt4o_model=models.get('gpt4o')
    )
    models['ensemble'] = ensemble
    
    return models


def evaluate_models(args, models, test_df):
    """Evaluate all models on the test set"""
    print("\n=== Evaluating Models ===")
    
    # Create sample for evaluation
    sample_df = sample_dataset(test_df, args.sample_size)
    
    # Track all metrics for comparison
    all_metrics = []
    
    # Evaluate each model
    for name, model in models.items():
        if name == 'ensemble':
            continue  # Evaluate ensemble after individual models
        
        print(f"\nEvaluating {name.upper()} model...")
        
        # Get predictions
        try:
            if name == 'gpt4o':
                # Use smaller sample for GPT-4o due to API costs
                gpt_sample_size = min(config.SAMPLE_SIZE['gpt4o'], len(sample_df))
                gpt_df = sample_df.sample(gpt_sample_size)
                results_df = model.batch_predict(gpt_df)
            else:
                results_df = model.batch_predict(sample_df)
            
            # Evaluate
            metrics = evaluate_model(results_df, name.upper())
            print_evaluation_report(metrics)
            
            # Save confusion matrix if requested
            if args.save_results:
                os.makedirs(args.output_dir, exist_ok=True)
                plot_confusion_matrix(
                    metrics, 
                    save_path=os.path.join(args.output_dir, f'{name}_confusion_matrix.png')
                )
            
            # Analyze errors
            error_analysis = analyze_errors(results_df, name.upper())
            print_error_analysis(error_analysis)
            
            # Save results if requested
            if args.save_results:
                results_df.to_csv(os.path.join(args.output_dir, f'{name}_results.csv'), index=False)
                
                # Visualize errors
                visualize_model_errors(error_analysis, sample_df, save_dir=args.output_dir)
            
            # Track metrics
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error evaluating {name} model: {e}")
    
    # Evaluate ensemble model if available
    if 'ensemble' in models:
        print("\nEvaluating ENSEMBLE model...")
        try:
            # Get ensemble predictions
            ensemble_results = models['ensemble'].batch_predict(sample_df)
            
            # Evaluate
            ensemble_metrics = evaluate_model(ensemble_results, 'ENSEMBLE')
            print_evaluation_report(ensemble_metrics)
            
            # Save confusion matrix if requested
            if args.save_results:
                plot_confusion_matrix(
                    ensemble_metrics, 
                    save_path=os.path.join(args.output_dir, 'ensemble_confusion_matrix.png')
                )
            
            # Analyze errors
            ensemble_error = analyze_errors(ensemble_results, 'ENSEMBLE')
            print_error_analysis(ensemble_error)
            
            # Save results if requested
            if args.save_results:
                ensemble_results.to_csv(os.path.join(args.output_dir, 'ensemble_results.csv'), index=False)
                
                # Visualize errors
                visualize_model_errors(ensemble_error, sample_df, save_dir=args.output_dir)
            
            # Add to metrics
            all_metrics.append(ensemble_metrics)
            
        except Exception as e:
            print(f"Error evaluating ensemble model: {e}")
    
    # Compare all models if we have multiple
    if len(all_metrics) > 1:
        print("\n=== Model Comparison ===")
        comparison = compare_models(all_metrics)
        print(comparison)
        
        # Plot comparison
        if args.save_results:
            for metric in ['Accuracy', 'AUC', 'F1 (Macro Avg)']:
                if metric in comparison.columns:
                    plot_model_comparison(
                        comparison, 
                        metric=metric,
                        save_path=os.path.join(args.output_dir, f'model_comparison_{metric}.png')
                    )
        
        # Plot ROC curves
        plot_roc_curve(
            all_metrics,
            save_path=os.path.join(args.output_dir, 'roc_curves.png') if args.save_results else None
        )
        
        # Plot precision-recall curves
        plot_precision_recall_curve(
            all_metrics,
            save_path=os.path.join(args.output_dir, 'pr_curves.png') if args.save_results else None
        )
    
    return all_metrics


def analyze_image(args, models, image_path):
    """Analyze a specific image with all models"""
    print(f"\n=== Analyzing Image: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Get predictions from each model
    predictions = {}
    
    for name, model in models.items():
        if name == 'ensemble':
            continue  # Skip ensemble for now
        
        print(f"Getting {name.upper()} prediction...")
        try:
            pred = model.predict(image_path)
            if pred is not None:
                predictions[name] = pred
        except Exception as e:
            print(f"Error with {name} model: {e}")
    
    # Get ensemble prediction if available
    if 'ensemble' in models and len(predictions) > 0:
        print("Getting ENSEMBLE prediction...")
        try:
            ensemble_pred = models['ensemble'].predict(image_path)
            if ensemble_pred is not None:
                predictions['ensemble'] = ensemble_pred
        except Exception as e:
            print(f"Error with ensemble model: {e}")
    
    # Visualize predictions
    if predictions:
        save_path = os.path.join(args.output_dir, 'image_analysis.png') if args.save_results else None
        visualize_image_with_predictions(image_path, predictions, save_path=save_path)
        
        # Also show detailed feature visualization
        feature_save_path = os.path.join(args.output_dir, 'image_features.png') if args.save_results else None
        visualize_image_with_features(image_path, save_path=feature_save_path)
    else:
        print("No valid predictions from any model")


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory if saving results
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data if requested
    if args.prepare_data:
        train_df, test_df = prepare_data(args)
    else:
        # Try to load existing dataset splits
        try:
            train_path = os.path.join(args.output_dir, 'train.csv')
            test_path = os.path.join(args.output_dir, 'test.csv')
            
            if os.path.exists(train_path) and os.path.exists(test_path):
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                print(f"Loaded existing dataset splits from {args.output_dir}")
            else:
                print("No existing dataset splits found. Creating new dataset...")
                train_df, test_df = prepare_data(args)
        except Exception as e:
            print(f"Error loading existing dataset: {e}")
            print("Creating new dataset...")
            train_df, test_df = prepare_data(args)
    
    # Train models
    models = train_models(args, train_df)
    
    # Evaluate models if requested
    if args.evaluate_models:
        evaluate_models(args, models, test_df)
    
    # Analyze specific image if requested
    if args.analyze_image:
        analyze_image(args, models, args.analyze_image)
    
    # Visualize samples if requested
    if args.visualize_samples:
        sample_df = sample_dataset(test_df, min(10, len(test_df)))
        visualize_sample_images(
            sample_df, 
            num_samples=5,
            save_dir=args.output_dir if args.save_results else None
        )


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")