"""
Evaluation utilities for the AI vs Real Image Detection project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import sys

# Add the project root to the path so we can import the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def evaluate_model(results_df, model_name):
    """
    Evaluate model performance
    
    Args:
        results_df: DataFrame with model predictions
        model_name: Name of the model
        
    Returns:
        dict: Evaluation metrics
    """
    # Check if we have predictions
    if 'is_real_prediction' not in results_df.columns:
        print(f"No predictions found for {model_name}")
        return None
    
    # Get actual and predicted labels
    y_true = results_df['actual_is_real']
    y_pred = results_df['is_real_prediction'].astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC curve and AUC if probability scores are available
    roc_data = None
    if 'real_probability' in results_df.columns:
        y_scores = results_df['real_probability']
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    # Calculate precision-recall curve if probability scores are available
    pr_data = None
    if 'real_probability' in results_df.columns:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
        pr_data = {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': average_precision
        }
    
    # Return all metrics
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_data': roc_data,
        'pr_data': pr_data
    }


def print_evaluation_report(metrics):
    """
    Print a readable evaluation report
    
    Args:
        metrics: Evaluation metrics from evaluate_model
    """
    if metrics is None:
        print("No metrics available")
        return
    
    print(f"\n===== {metrics['model_name']} Evaluation =====")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Print classification report
    report = metrics['classification_report']
    print("\nClassification Report:")
    print(f"              precision    recall  f1-score   support")
    print(f"           0      {report['0']['precision']:.4f}     {report['0']['recall']:.4f}     {report['0']['f1-score']:.4f}     {report['0']['support']}")
    print(f"           1      {report['1']['precision']:.4f}     {report['1']['recall']:.4f}     {report['1']['f1-score']:.4f}     {report['1']['support']}")
    print(f"    accuracy                          {report['accuracy']:.4f}    {report['macro avg']['support']}")
    print(f"   macro avg      {report['macro avg']['precision']:.4f}     {report['macro avg']['recall']:.4f}     {report['macro avg']['f1-score']:.4f}     {report['macro avg']['support']}")
    print(f"weighted avg      {report['weighted avg']['precision']:.4f}     {report['weighted avg']['recall']:.4f}     {report['weighted avg']['f1-score']:.4f}     {report['weighted avg']['support']}")
    
    # Print AUC if available
    if metrics['roc_data'] is not None:
        print(f"\nROC AUC: {metrics['roc_data']['auc']:.4f}")
    
    # Print average precision if available
    if metrics['pr_data'] is not None:
        print(f"Average Precision: {metrics['pr_data']['average_precision']:.4f}")


def plot_confusion_matrix(metrics, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        metrics: Evaluation metrics from evaluate_model
        save_path: Optional path to save the figure
    """
    if metrics is None or 'confusion_matrix' not in metrics:
        print("No confusion matrix available")
        return
    
    cm = metrics['confusion_matrix']
    model_name = metrics['model_name']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(metrics_list, save_path=None):
    """
    Plot ROC curves for multiple models
    
    Args:
        metrics_list: List of evaluation metrics from evaluate_model
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Add ROC curve for each model
    for metrics in metrics_list:
        if metrics is None or 'roc_data' not in metrics or metrics['roc_data'] is None:
            continue
        
        model_name = metrics['model_name']
        roc_data = metrics['roc_data']
        
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                label=f"{model_name} (AUC = {roc_data['auc']:.4f})")
    
    # Add random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(metrics_list, save_path=None):
    """
    Plot precision-recall curves for multiple models
    
    Args:
        metrics_list: List of evaluation metrics from evaluate_model
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Add precision-recall curve for each model
    for metrics in metrics_list:
        if metrics is None or 'pr_data' not in metrics or metrics['pr_data'] is None:
            continue
        
        model_name = metrics['model_name']
        pr_data = metrics['pr_data']
        
        plt.plot(pr_data['recall'], pr_data['precision'], 
                label=f"{model_name} (AP = {pr_data['average_precision']:.4f})")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved precision-recall curves to {save_path}")
    
    plt.show()


def compare_models(metrics_list):
    """
    Compare model performances
    
    Args:
        metrics_list: List of evaluation metrics from evaluate_model
        
    Returns:
        pandas.DataFrame: Comparison of model metrics
    """
    # Extract key metrics for each model
    comparison = []
    
    for metrics in metrics_list:
        if metrics is None:
            continue
        
        model_data = {
            'Model': metrics['model_name'],
            'Accuracy': metrics['accuracy']
        }
        
        # Add AUC if available
        if 'roc_data' in metrics and metrics['roc_data'] is not None:
            model_data['AUC'] = metrics['roc_data']['auc']
        else:
            model_data['AUC'] = np.nan
        
        # Add average precision if available
        if 'pr_data' in metrics and metrics['pr_data'] is not None:
            model_data['AP'] = metrics['pr_data']['average_precision']
        else:
            model_data['AP'] = np.nan
        
        # Add F1 scores from classification report
        if 'classification_report' in metrics:
            report = metrics['classification_report']
            model_data['F1 (Fake)'] = report['0']['f1-score']
            model_data['F1 (Real)'] = report['1']['f1-score']
            model_data['F1 (Macro Avg)'] = report['macro avg']['f1-score']
        
        comparison.append(model_data)
    
    # Convert to dataframe
    comparison_df = pd.DataFrame(comparison)
    
    return comparison_df


def plot_model_comparison(comparison_df, metric='Accuracy', save_path=None):
    """
    Plot comparison of model performances for a specific metric
    
    Args:
        comparison_df: DataFrame from compare_models
        metric: Metric to compare
        save_path: Optional path to save the figure
    """
    if metric not in comparison_df.columns:
        print(f"Metric '{metric}' not found in comparison data")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort by the metric
    sorted_df = comparison_df.sort_values(metric, ascending=False)
    
    # Create bar plot
    ax = sns.barplot(x='Model', y=metric, data=sorted_df)
    
    # Add value labels on top of each bar
    for i, v in enumerate(sorted_df[metric]):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison - {metric}')
    plt.ylabel(metric)
    plt.ylim(0, min(1.1, sorted_df[metric].max() + 0.1))
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
    
    plt.tight_layout()
    plt.show()


def analyze_errors(results_df, model_name):
    """
    Analyze errors made by the model
    
    Args:
        results_df: DataFrame with model predictions
        model_name: Name of the model
        
    Returns:
        dict: Error analysis results
    """
    # Check for required columns
    required_cols = ['is_real_prediction', 'actual_is_real', 'image_path']
    if not all(col in results_df.columns for col in required_cols):
        print(f"Missing required columns for error analysis")
        return None
    
    # Find errors
    errors = results_df[results_df['is_real_prediction'] != results_df['actual_is_real']]
    
    # Categorize errors
    false_positives = errors[errors['actual_is_real'] == 0]  # Fake images predicted as real
    false_negatives = errors[errors['actual_is_real'] == 1]  # Real images predicted as fake
    
    # Calculate error rates
    total_real = len(results_df[results_df['actual_is_real'] == 1])
    total_fake = len(results_df[results_df['actual_is_real'] == 0])
    
    fp_rate = len(false_positives) / total_fake if total_fake > 0 else 0
    fn_rate = len(false_negatives) / total_real if total_real > 0 else 0
    
    # Get image paths for detailed inspection
    fp_images = false_positives['image_path'].tolist()
    fn_images = false_negatives['image_path'].tolist()
    
    # Return analysis
    return {
        'model_name': model_name,
        'total_errors': len(errors),
        'error_rate': len(errors) / len(results_df),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'fp_images': fp_images,
        'fn_images': fn_images
    }


def print_error_analysis(error_analysis):
    """
    Print error analysis results
    
    Args:
        error_analysis: Results from analyze_errors
    """
    if error_analysis is None:
        print("No error analysis available")
        return
    
    print(f"\n===== {error_analysis['model_name']} Error Analysis =====")
    print(f"Total errors: {error_analysis['total_errors']} ({error_analysis['error_rate']:.2%} of test set)")
    print(f"False positives (Fake as Real): {error_analysis['false_positives']} ({error_analysis['fp_rate']:.2%} of fake images)")
    print(f"False negatives (Real as Fake): {error_analysis['false_negatives']} ({error_analysis['fn_rate']:.2%} of real images)")
    
    # Print sample of error images
    if error_analysis['fp_images']:
        print("\nSample false positive images (up to 5):")
        for img in error_analysis['fp_images'][:5]:
            print(f" - {img}")
    
    if error_analysis['fn_images']:
        print("\nSample false negative images (up to 5):")
        for img in error_analysis['fn_images'][:5]:
            print(f" - {img}")


def analyze_confidence_distribution(results_df, model_name):
    """
    Analyze the distribution of model confidence
    
    Args:
        results_df: DataFrame with model predictions
        model_name: Name of the model
        
    Returns:
        dict: Confidence analysis
    """
    # Check for required columns
    if 'confidence' not in results_df.columns or 'actual_is_real' not in results_df.columns:
        print(f"Missing required columns for confidence analysis")
        return None
    
    # Calculate average confidence
    avg_confidence = results_df['confidence'].mean()
    
    # Calculate average confidence for correct and incorrect predictions
    correct = results_df['is_real_prediction'] == results_df['actual_is_real']
    avg_confidence_correct = results_df[correct]['confidence'].mean()
    avg_confidence_incorrect = results_df[~correct]['confidence'].mean() if (~correct).any() else np.nan
    
    # Calculate confidence by actual class
    real_images = results_df['actual_is_real'] == 1
    fake_images = results_df['actual_is_real'] == 0
    
    avg_confidence_real = results_df[real_images]['confidence'].mean() if real_images.any() else np.nan
    avg_confidence_fake = results_df[fake_images]['confidence'].mean() if fake_images.any() else np.nan
    
    return {
        'model_name': model_name,
        'avg_confidence': avg_confidence,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect,
        'avg_confidence_real': avg_confidence_real,
        'avg_confidence_fake': avg_confidence_fake
    }


def plot_confidence_distribution(results_df, model_name, save_path=None):
    """
    Plot the distribution of model confidence
    
    Args:
        results_df: DataFrame with model predictions
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    # Check for required columns
    if 'confidence' not in results_df.columns or 'actual_is_real' not in results_df.columns:
        print(f"Missing required columns for confidence analysis")
        return
    
    # Create correct/incorrect classification
    results_df['correct'] = results_df['is_real_prediction'] == results_df['actual_is_real']
    results_df['prediction_type'] = results_df.apply(
        lambda row: 'Correct' if row['correct'] else 'Incorrect', axis=1
    )
    
    plt.figure(figsize=(12, 6))
    
    # Plot confidence distribution by correctness
    sns.histplot(data=results_df, x='confidence', hue='prediction_type', 
                 bins=20, kde=True, alpha=0.5)
    
    plt.title(f'{model_name} Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confidence distribution to {save_path}")
    
    plt.show()
    
    # Also plot confidence by actual class
    plt.figure(figsize=(12, 6))
    
    sns.histplot(data=results_df, x='confidence', hue='actual_label',
                 bins=20, kde=True, alpha=0.5)
    
    plt.title(f'{model_name} Confidence by Class')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    
    if save_path:
        class_save_path = save_path.replace('.', '_by_class.')
        plt.savefig(class_save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confidence by class to {class_save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test evaluation functions
    from utils.dataset import create_dataset
    from models.clip_model import CLIPClassifier
    
    # Create dataset
    df = create_dataset()
    
    # Test with a small sample
    test_df = df.sample(10)
    
    # Create synthetic results for testing
    results = {
        'image_path': test_df['path'].tolist(),
        'actual_label': test_df['label'].tolist(),
        'actual_is_real': test_df['is_real'].tolist(),
        'is_real_prediction': [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        'real_probability': [0.8, 0.3, 0.9, 0.7, 0.4, 0.6, 0.9, 0.2, 0.3, 0.8],
        'fake_probability': [0.2, 0.7, 0.1, 0.3, 0.6, 0.4, 0.1, 0.8, 0.7, 0.2],
        'confidence': [0.8, 0.7, 0.9, 0.7, 0.6, 0.6, 0.9, 0.8, 0.7, 0.8]
    }
    
    results_df = pd.DataFrame(results)
    
    # Evaluate
    metrics = evaluate_model(results_df, 'Test Model')
    print_evaluation_report(metrics)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics)
    
    # Test error analysis
    error_analysis = analyze_errors(results_df, 'Test Model')
    print_error_analysis(error_analysis)
    
    # Test confidence analysis
    plot_confidence_distribution(results_df, 'Test Model')