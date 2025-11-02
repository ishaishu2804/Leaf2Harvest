"""
Quick Model Evaluation Script - Fast Results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

print("🚀 QUICK MODEL EVALUATION - FAST RESULTS")
print("="*50)

# Load data quickly
print("📊 Loading data...")
df = pd.read_csv('data/Crop_recommendation.csv')
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

# Train models quickly
print("\n🤖 Training models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500)
}

# Load existing model
try:
    existing_model = joblib.load('app/models/crop_model.pkl')
    models['Existing Model'] = existing_model
    print("✅ Loaded existing trained model")
except:
    print("⚠️  No existing model found")

# Train and evaluate
results = []
for name, model in models.items():
    if name != 'Existing Model':
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}"
    })
    
    print(f"  ✅ {name}: {accuracy:.4f} accuracy")

# Display results table
print("\n📋 RESULTS TABLE")
print("="*80)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))
print("="*80)

# Quick visualization
print("\n📊 Creating visualizations...")
plt.figure(figsize=(10, 6))
models_list = results_df['Model'].tolist()
accuracies = [float(acc) for acc in results_df['Accuracy'].tolist()]

bars = plt.bar(models_list, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.ylim(0, 1.0)

# Add values on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('quick_accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Chart saved as 'quick_accuracy_comparison.png'")

# Confusion matrix for best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('quick_confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"✅ Confusion matrix saved for {best_model_name}")

print("\n🎉 EVALUATION COMPLETE!")
print(f"🏆 Best Model: {best_model_name} with {results_df.iloc[0]['Accuracy']} accuracy")
print("📁 Files created: quick_accuracy_comparison.png, quick_confusion_matrix.png")
