#!/usr/bin/env python3
"""
Script to run GPT-4o evaluation on the full sample with proper rate limiting
"""

import os
import pandas as pd
import time
import sys
import random
from utils.dataset import sample_dataset
from utils.evaluation import evaluate_model, print_evaluation_report, plot_confusion_matrix
from models.gpt4o_model import GPT4oClassifier
import config

# Load test dataset
test_path = os.path.join(config.RESULTS_DIR, 'test.csv')
test_df = pd.read_csv(test_path)

# Use the full test set
print(f"Using full test dataset with {len(test_df)} images")

# Initialize GPT-4o model
print("\n=== Initializing GPT-4o Model ===")
gpt4o_model = GPT4oClassifier(api_key=config.GPT4O_API_KEY)
print("GPT-4o model initialized successfully")

# Check for existing partial results to resume from
partial_results_path = os.path.join(config.RESULTS_DIR, 'gpt4o_partial_results.csv')
processed_paths = []
results = []

if os.path.exists(partial_results_path):
    try:
        partial_df = pd.read_csv(partial_results_path)
        results = partial_df.to_dict('records')
        processed_paths = [r['image_path'] for r in results]
        print(f"Resuming from partial results: {len(processed_paths)} images already processed")
    except Exception as e:
        print(f"Error loading partial results, starting fresh: {e}")

# Set delay between requests to avoid rate limiting
min_delay = 8   # seconds
max_delay = 12  # seconds

# Set batch size and delay between batches
batch_size = 3
batch_delay = 30  # seconds

# Process images in batches
print("\n=== Evaluating GPT-4o Model ===")
start_time = time.time()

# Filter out already processed images
remaining_df = test_df[~test_df['path'].isin(processed_paths)]
print(f"Remaining images to process: {len(remaining_df)}")

# Process in batches
for i in range(0, len(remaining_df), batch_size):
    batch_df = remaining_df.iloc[i:i+batch_size]
    print(f"\nProcessing batch {i//batch_size + 1}/{(len(remaining_df) + batch_size - 1)//batch_size}")
    
    for idx, row in batch_df.iterrows():
        print(f"Processing image {len(processed_paths) + 1}/{len(test_df)}: {os.path.basename(row['path'])}")
        
        try:
            # Get prediction
            pred = gpt4o_model.classify_image(row['path'])
            
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
            processed_paths.append(row['path'])
            
            print(f"Result: {'REAL' if pred['is_real_prediction'] else 'FAKE'} (confidence: {pred['confidence']:.2f})")
            
            # Save intermediate results after each image
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_csv(partial_results_path, index=False)
            
            # Only sleep if there are more images to process in this batch
            if idx < batch_df.index[-1]:
                delay = random.uniform(min_delay, max_delay)
                print(f"Sleeping for {delay:.1f} seconds before next image...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"Error processing image {row['path']}: {e}")
            # Continue with next image instead of failing entirely
    
    # Sleep between batches if there are more batches to process
    if i + batch_size < len(remaining_df):
        print(f"Batch complete. Sleeping for {batch_delay} seconds before next batch...")
        time.sleep(batch_delay)

# Convert all results to dataframe
results_df = pd.DataFrame(results)

# Save final results
if not results_df.empty:
    results_path = os.path.join(config.RESULTS_DIR, 'gpt4o_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Evaluate
    try:
        metrics = evaluate_model(results_df, "GPT4O")
        print_evaluation_report(metrics)
        
        # Save confusion matrix
        cm_path = os.path.join(config.RESULTS_DIR, 'gpt4o_confusion_matrix.png')
        plot_confusion_matrix(metrics, save_path=cm_path)
    except Exception as e:
        print(f"Error evaluating results: {e}")
        
    # Print summary
    print("\nGPT-4o Results Summary:")
    print(f"Total images processed: {len(results_df)}")
    print(f"Real images: {len(results_df[results_df['actual_is_real'] == 1])}")
    print(f"Fake images: {len(results_df[results_df['actual_is_real'] == 0])}")
    
    # Count valid predictions
    valid_preds = results_df[results_df['is_real_prediction'].notna()]
    if not valid_preds.empty:
        correct = valid_preds['is_real_prediction'] == valid_preds['actual_is_real']
        accuracy = correct.mean()
        print(f"Valid predictions: {len(valid_preds)}")
        print(f"Correct predictions: {sum(correct)}")
        print(f"Accuracy: {accuracy:.2%}")
else:
    print("No valid results obtained")

elapsed_time = time.time() - start_time
print(f"\nGPT-4o evaluation completed in {elapsed_time:.2f} seconds")