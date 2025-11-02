"""
Machine Learning Model Performance Evaluation and Visualization Script

This script evaluates and visualizes the performance of multiple machine learning models
including Random Forest, SVM, CNN, and Logistic Regression models for agricultural applications.

Author: LeafToHarvest Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """
    A comprehensive class for evaluating and visualizing machine learning model performance.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator with empty results storage."""
        self.results = {}
        self.models = {}
        self.X_test = None
        self.y_test = None
        
    def load_crop_data(self):
        """
        Load and prepare crop recommendation dataset.
        Returns features (X) and target (y) for crop prediction.
        """
        print("[INFO] Loading crop recommendation dataset...")
        df = pd.read_csv('data/Crop_recommendation.csv')
        
        # Prepare features and target
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"[INFO] Crop dataset loaded. Shape: {X.shape}")
        print(f"[INFO] Test set size: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def load_pest_data(self):
        """
        Load and prepare pest risk dataset.
        Returns features (X) and target (y) for pest prediction.
        """
        print("[INFO] Loading pest risk dataset...")
        df = pd.read_csv('data/pest_data.csv')
        
        # Encode crop type
        le = LabelEncoder()
        df['crop_type'] = le.fit_transform(df['crop_type'])
        
        # Prepare features and target
        X = df[['temperature', 'humidity', 'rainfall', 'crop_type']]
        y = df['pest_risk']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[INFO] Pest dataset loaded. Shape: {X.shape}")
        print(f"[INFO] Test set size: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train, dataset_type='crop'):
        """
        Train multiple models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            dataset_type: Type of dataset ('crop' or 'pest')
        """
        print(f"[INFO] Training models for {dataset_type} prediction...")
        
        # Random Forest
        print("[INFO] Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        # Support Vector Machine
        print("[INFO] Training SVM...")
        svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        # Logistic Regression
        print("[INFO] Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        # Load existing trained models if available
        self.load_existing_models(dataset_type)
        
        print(f"[INFO] All models trained/loaded for {dataset_type} prediction.")
    
    def load_existing_models(self, dataset_type):
        """
        Load existing trained models from the models directory.
        
        Args:
            dataset_type: Type of dataset ('crop' or 'pest')
        """
        try:
            if dataset_type == 'crop' and os.path.exists('app/models/crop_model.pkl'):
                print("[INFO] Loading existing crop model...")
                existing_model = joblib.load('app/models/crop_model.pkl')
                self.models['Existing Crop Model'] = existing_model
                
            elif dataset_type == 'pest' and os.path.exists('app/models/pest_model.pkl'):
                print("[INFO] Loading existing pest model...")
                existing_model = joblib.load('app/models/pest_model.pkl')
                self.models['Existing Pest Model'] = existing_model
                
        except Exception as e:
            print(f"[WARNING] Could not load existing model: {e}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and calculate performance metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("[INFO] Evaluating models...")
        
        for model_name, model in self.models.items():
            print(f"[INFO] Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            self.results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Predictions': y_pred
            }
            
            print(f"[INFO] {model_name} - Accuracy: {accuracy:.4f}")
    
    def create_results_table(self):
        """
        Create a pandas DataFrame with model performance results.
        
        Returns:
            pd.DataFrame: Results table with metrics for each model
        """
        print("[INFO] Creating results table...")
        
        # Prepare data for DataFrame
        data = []
        for model_name, metrics in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1-Score': f"{metrics['F1-Score']:.4f}"
            })
        
        results_df = pd.DataFrame(data)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        return results_df
    
    def plot_accuracy_comparison(self, save_path='model_accuracy_comparison.png'):
        """
        Create a bar chart comparing model accuracies.
        
        Args:
            save_path: Path to save the plot
        """
        print("[INFO] Creating accuracy comparison chart...")
        
        # Prepare data for plotting
        models = list(self.results.keys())
        accuracies = [self.results[model]['Accuracy'] for model in models]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        # Customize the plot
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.0)
        
        # Add accuracy values on top of bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Accuracy comparison chart saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, model_name='Random Forest', save_path='confusion_matrix.png'):
        """
        Create a confusion matrix heatmap for the specified model.
        
        Args:
            model_name: Name of the model to create confusion matrix for
            save_path: Path to save the plot
        """
        print(f"[INFO] Creating confusion matrix for {model_name}...")
        
        if model_name not in self.results:
            print(f"[ERROR] Model '{model_name}' not found in results.")
            return
        
        # Get predictions and true labels
        y_pred = self.results[model_name]['Predictions']
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(self.y_test), 
                   yticklabels=np.unique(self.y_test))
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, model_name='Random Forest'):
        """
        Generate a detailed classification report for a specific model.
        
        Args:
            model_name: Name of the model to generate report for
        """
        print(f"[INFO] Generating detailed report for {model_name}...")
        
        if model_name not in self.results:
            print(f"[ERROR] Model '{model_name}' not found in results.")
            return
        
        y_pred = self.results[model_name]['Predictions']
        
        print(f"\n{'='*60}")
        print(f"DETAILED CLASSIFICATION REPORT - {model_name.upper()}")
        print(f"{'='*60}")
        print(classification_report(self.y_test, y_pred))
        print(f"{'='*60}")

def main():
    """
    Main function to run the complete model evaluation pipeline.
    """
    print("="*80)
    print("MACHINE LEARNING MODEL EVALUATION AND VISUALIZATION")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Choose dataset type (crop or pest)
    dataset_type = 'crop'  # Change to 'pest' for pest risk evaluation
    
    try:
        # Load data based on dataset type
        if dataset_type == 'crop':
            X_train, X_test, y_train, y_test = evaluator.load_crop_data()
        else:
            X_train, X_test, y_train, y_test = evaluator.load_pest_data()
        
        # Train models
        evaluator.train_models(X_train, y_train, dataset_type)
        
        # Evaluate models
        evaluator.evaluate_models(X_test, y_test)
        
        # Create and display results table
        results_df = evaluator.create_results_table()
        
        # Create visualizations
        evaluator.plot_accuracy_comparison()
        evaluator.plot_confusion_matrix()
        
        # Generate detailed report for the best model
        best_model = results_df.iloc[0]['Model']
        evaluator.generate_detailed_report(best_model)
        
        print("\n[INFO] Model evaluation completed successfully!")
        print("[INFO] Check the generated plots and results table above.")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
