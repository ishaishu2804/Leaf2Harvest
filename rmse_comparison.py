"""
RMSE Comparison Plot Generator for Machine Learning Models
Creates RMSE comparison visualization similar to accuracy bar plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("📊 RMSE COMPARISON PLOT GENERATOR")
print("="*50)

def load_and_prepare_data():
    """Load and prepare data for RMSE calculation"""
    print("📊 Loading crop recommendation dataset...")
    df = pd.read_csv('data/Crop_recommendation.csv')
    
    # Prepare features
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    
    # Encode target labels for regression (convert to numeric)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['label'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Test set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, le

def train_regression_models(X_train, y_train):
    """Train regression models for RMSE comparison"""
    print("\n🤖 Training regression models...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'SVM': SVR(kernel='rbf'),
        'Linear Regression': LinearRegression()
    }
    
    # Load existing model and convert to regression
    try:
        existing_model = joblib.load('app/models/crop_model.pkl')
        # Convert classifier to regressor by using predict_proba
        models['Existing Model (Converted)'] = existing_model
        print("✅ Loaded existing model (converted for regression)")
    except:
        print("⚠️  No existing model found")
    
    # Train models
    for name, model in models.items():
        if name != 'Existing Model (Converted)':
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
        else:
            print(f"  Using {name}...")
    
    return models

def calculate_rmse_scores(models, X_test, y_test):
    """Calculate RMSE scores for all models"""
    print("\n📈 Calculating RMSE scores...")
    
    results = []
    
    for name, model in models.items():
        if name == 'Existing Model (Converted)':
            # For classifier, use predict_proba and get class probabilities
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append({
            'Model': name,
            'RMSE': rmse
        })
        
        print(f"  ✅ {name}: RMSE = {rmse:.4f}")
    
    return results

def create_rmse_comparison_plot(results, save_path='rmse_comparison.png'):
    """Create RMSE comparison bar plot"""
    print("\n📊 Creating RMSE comparison plot...")
    
    # Prepare data for plotting
    models = [result['Model'] for result in results]
    rmse_scores = [result['RMSE'] for result in results]
    
    # Sort by RMSE (ascending - lower is better)
    sorted_data = sorted(zip(models, rmse_scores), key=lambda x: x[1])
    models, rmse_scores = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(models, rmse_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # Customize the plot
    plt.title('Model RMSE Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Machine Learning Models', fontsize=12, fontweight='bold')
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
    
    # Add RMSE values on top of bars
    for bar, rmse in zip(bars, rmse_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rmse:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add note about lower RMSE being better
    plt.text(0.02, 0.98, 'Note: Lower RMSE indicates better performance', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ RMSE comparison plot saved to {save_path}")
    
    plt.show()

def create_rmse_table(results):
    """Create and display RMSE results table"""
    print("\n📋 RMSE RESULTS TABLE")
    print("="*60)
    
    # Sort by RMSE (ascending - lower is better)
    results_sorted = sorted(results, key=lambda x: x['RMSE'])
    
    # Create DataFrame
    df = pd.DataFrame(results_sorted)
    df['RMSE'] = df['RMSE'].round(4)
    
    print(df.to_string(index=False))
    print("="*60)
    
    # Show best model
    best_model = results_sorted[0]
    print(f"\n🏆 Best Model: {best_model['Model']} with RMSE = {best_model['RMSE']:.4f}")
    print("💡 Lower RMSE indicates better performance")

def main():
    """Main function to generate RMSE comparison"""
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, le = load_and_prepare_data()
        
        # Train models
        models = train_regression_models(X_train, y_train)
        
        # Calculate RMSE scores
        results = calculate_rmse_scores(models, X_test, y_test)
        
        # Create results table
        create_rmse_table(results)
        
        # Create RMSE comparison plot
        create_rmse_comparison_plot(results)
        
        print("\n🎉 RMSE comparison completed successfully!")
        print("📁 Files created: rmse_comparison.png")
        
    except Exception as e:
        print(f"❌ Error during RMSE calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
