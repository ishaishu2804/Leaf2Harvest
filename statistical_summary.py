"""
Statistical Summary Generator for LeafToHarvest Datasets
Generates clean statistical summaries for research papers
"""

import pandas as pd
import numpy as np

print("📊 STATISTICAL SUMMARY GENERATOR")
print("="*60)

def generate_summary(dataset_path, dataset_name):
    """
    Generate statistical summary for a dataset
    
    Args:
        dataset_path: Path to the CSV file
        dataset_name: Name of the dataset for display
    """
    print(f"\n🔍 Analyzing {dataset_name}...")
    
    try:
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Select only numeric columns for statistical analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_data = data[numeric_cols]
        
        # Generate statistical summary
        summary = numeric_data.describe().T
        
        # Round values for cleaner presentation
        summary = summary[['max', 'min', 'mean', 'std']].round(2)
        
        # Rename columns to match research paper format
        summary.columns = ['Maximum value', 'Minimum value', 'Mean', 'Standard deviation']
        
        print(f"\n📋 STATISTICAL SUMMARY - {dataset_name.upper()}")
        print("="*80)
        print(f"Dataset Shape: {data.shape[0]} samples, {data.shape[1]} features")
        print(f"Numeric Features: {len(numeric_cols)}")
        print("-"*80)
        print(summary.to_string())
        print("="*80)
        
        # Additional insights
        print(f"\n💡 KEY INSIGHTS - {dataset_name}:")
        print(f"   • Total samples: {data.shape[0]:,}")
        print(f"   • Features analyzed: {len(numeric_cols)}")
        print(f"   • Missing values: {data.isnull().sum().sum()}")
        print(f"   • Data types: {data.dtypes.value_counts().to_dict()}")
        
        return summary
        
    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}: {e}")
        return None

def main():
    """Main function to generate statistical summaries"""
    
    # Analyze Crop Recommendation Dataset
    crop_summary = generate_summary("data/Crop_recommendation.csv", "Crop Recommendation")
    
    # Analyze Pest Data Dataset
    pest_summary = generate_summary("data/pest_data.csv", "Pest Risk Assessment")
    
    # Generate combined insights
    print("\n🎯 COMBINED DATASET INSIGHTS")
    print("="*60)
    
    # Load both datasets for comparison
    try:
        crop_data = pd.read_csv("data/Crop_recommendation.csv")
        pest_data = pd.read_csv("data/pest_data.csv")
        
        print(f"📈 Crop Dataset: {crop_data.shape[0]:,} samples, {crop_data.shape[1]} features")
        print(f"🐛 Pest Dataset: {pest_data.shape[0]:,} samples, {pest_data.shape[1]} features")
        print(f"📊 Total samples across both datasets: {crop_data.shape[0] + pest_data.shape[0]:,}")
        
        # Show unique values in target columns
        print(f"\n🌾 Crop types in dataset: {crop_data['label'].nunique()} unique crops")
        print(f"🔍 Pest risk levels: {pest_data['pest_risk'].nunique()} levels")
        
        # Show sample of unique values
        print(f"\n📝 Sample crop types: {list(crop_data['label'].unique()[:10])}")
        print(f"📝 Pest risk levels: {list(pest_data['pest_risk'].unique())}")
        
    except Exception as e:
        print(f"❌ Error in combined analysis: {e}")
    
    print("\n✅ Statistical summary generation completed!")
    print("📁 Use these summaries in your research paper or documentation.")

if __name__ == "__main__":
    main()
