"""
Scatter Plot Analysis for Agricultural Risk Assessment
Creates scatter plots showing relationships between risk probability and environmental factors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("📊 SCATTER PLOT ANALYSIS GENERATOR")
print("="*50)

def load_and_prepare_data():
    """Load and prepare data for scatter plot analysis"""
    print("📊 Loading crop recommendation dataset...")
    
    # Load the crop dataset
    df = pd.read_csv('data/Crop_recommendation.csv')
    
    # Create risk probability based on environmental factors
    # Higher temperature + humidity + rainfall = higher risk
    df['Risk_Probability'] = (
        (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min()) * 0.3 +
        (df['humidity'] - df['humidity'].min()) / (df['humidity'].max() - df['humidity'].min()) * 0.3 +
        (df['rainfall'] - df['rainfall'].min()) / (df['rainfall'].max() - df['rainfall'].min()) * 0.4
    )
    
    # Add some noise to make it more realistic
    np.random.seed(42)
    df['Risk_Probability'] += np.random.normal(0, 0.05, len(df))
    df['Risk_Probability'] = np.clip(df['Risk_Probability'], 0, 1)
    
    print(f"✅ Data loaded: {df.shape[0]} samples")
    print(f"✅ Risk probability range: {df['Risk_Probability'].min():.3f} - {df['Risk_Probability'].max():.3f}")
    
    return df

def create_risk_vs_rainfall_plot(df, save_path='risk_vs_rainfall.png'):
    """Create scatter plot: Risk Probability vs Rainfall"""
    print("🌧️ Creating Risk vs Rainfall scatter plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color gradient
    scatter = plt.scatter(df['rainfall'], df['Risk_Probability'], 
                         c=df['Risk_Probability'], cmap='RdYlBu_r', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df['rainfall'], df['Risk_Probability'], 1)
    p = np.poly1d(z)
    plt.plot(df['rainfall'], p(df['rainfall']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate R²
    r2 = r2_score(df['Risk_Probability'], p(df['rainfall']))
    
    # Customize plot
    plt.xlabel('Rainfall (mm)', fontsize=14, fontweight='bold')
    plt.ylabel('Risk Probability', fontsize=14, fontweight='bold')
    plt.title(f'Agricultural Risk Probability vs Rainfall\nR² = {r2:.3f}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Risk Probability', fontsize=12, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['rainfall'].corr(df['Risk_Probability'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Risk vs Rainfall plot saved to {save_path}")
    plt.show()

def create_risk_vs_temperature_plot(df, save_path='risk_vs_temperature.png'):
    """Create scatter plot: Risk Probability vs Temperature"""
    print("🌡️ Creating Risk vs Temperature scatter plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color gradient
    scatter = plt.scatter(df['temperature'], df['Risk_Probability'], 
                         c=df['Risk_Probability'], cmap='RdYlBu_r', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df['temperature'], df['Risk_Probability'], 1)
    p = np.poly1d(z)
    plt.plot(df['temperature'], p(df['temperature']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate R²
    r2 = r2_score(df['Risk_Probability'], p(df['temperature']))
    
    # Customize plot
    plt.xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
    plt.ylabel('Risk Probability', fontsize=14, fontweight='bold')
    plt.title(f'Agricultural Risk Probability vs Temperature\nR² = {r2:.3f}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Risk Probability', fontsize=12, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['temperature'].corr(df['Risk_Probability'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Risk vs Temperature plot saved to {save_path}")
    plt.show()

def create_risk_vs_humidity_plot(df, save_path='risk_vs_humidity.png'):
    """Create scatter plot: Risk Probability vs Humidity"""
    print("💧 Creating Risk vs Humidity scatter plot...")
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color gradient
    scatter = plt.scatter(df['humidity'], df['Risk_Probability'], 
                         c=df['Risk_Probability'], cmap='RdYlBu_r', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df['humidity'], df['Risk_Probability'], 1)
    p = np.poly1d(z)
    plt.plot(df['humidity'], p(df['humidity']), "r--", alpha=0.8, linewidth=2)
    
    # Calculate R²
    r2 = r2_score(df['Risk_Probability'], p(df['humidity']))
    
    # Customize plot
    plt.xlabel('Humidity (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Risk Probability', fontsize=14, fontweight='bold')
    plt.title(f'Agricultural Risk Probability vs Humidity\nR² = {r2:.3f}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Risk Probability', fontsize=12, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['humidity'].corr(df['Risk_Probability'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Risk vs Humidity plot saved to {save_path}")
    plt.show()

def create_multi_factor_plot(df, save_path='multi_factor_analysis.png'):
    """Create multi-factor scatter plot analysis"""
    print("🔍 Creating multi-factor analysis plot...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Risk vs Rainfall
    ax1.scatter(df['rainfall'], df['Risk_Probability'], alpha=0.6, s=30, c='blue')
    z1 = np.polyfit(df['rainfall'], df['Risk_Probability'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(df['rainfall'], p1(df['rainfall']), "r--", alpha=0.8)
    ax1.set_xlabel('Rainfall (mm)')
    ax1.set_ylabel('Risk Probability')
    ax1.set_title(f'Risk vs Rainfall\nR² = {r2_score(df["Risk_Probability"], p1(df["rainfall"])):.3f}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk vs Temperature
    ax2.scatter(df['temperature'], df['Risk_Probability'], alpha=0.6, s=30, c='red')
    z2 = np.polyfit(df['temperature'], df['Risk_Probability'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['temperature'], p2(df['temperature']), "r--", alpha=0.8)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Risk Probability')
    ax2.set_title(f'Risk vs Temperature\nR² = {r2_score(df["Risk_Probability"], p2(df["temperature"])):.3f}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Risk vs Humidity
    ax3.scatter(df['humidity'], df['Risk_Probability'], alpha=0.6, s=30, c='green')
    z3 = np.polyfit(df['humidity'], df['Risk_Probability'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(df['humidity'], p3(df['humidity']), "r--", alpha=0.8)
    ax3.set_xlabel('Humidity (%)')
    ax3.set_ylabel('Risk Probability')
    ax3.set_title(f'Risk vs Humidity\nR² = {r2_score(df["Risk_Probability"], p3(df["humidity"])):.3f}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk vs pH
    ax4.scatter(df['ph'], df['Risk_Probability'], alpha=0.6, s=30, c='orange')
    z4 = np.polyfit(df['ph'], df['Risk_Probability'], 1)
    p4 = np.poly1d(z4)
    ax4.plot(df['ph'], p4(df['ph']), "r--", alpha=0.8)
    ax4.set_xlabel('pH Level')
    ax4.set_ylabel('Risk Probability')
    ax4.set_title(f'Risk vs pH\nR² = {r2_score(df["Risk_Probability"], p4(df["ph"])):.3f}')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Factor Agricultural Risk Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Multi-factor analysis saved to {save_path}")
    plt.show()

def create_correlation_heatmap(df, save_path='correlation_heatmap.png'):
    """Create correlation heatmap for all factors"""
    print("🔥 Creating correlation heatmap...")
    
    # Select relevant columns
    cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Risk_Probability']
    corr_data = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Matrix: Environmental Factors vs Risk Probability', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation heatmap saved to {save_path}")
    plt.show()

def generate_insights(df):
    """Generate insights from the scatter plot analysis"""
    print("\n💡 SCATTER PLOT ANALYSIS INSIGHTS")
    print("="*60)
    
    # Calculate correlations
    correlations = {
        'Rainfall': df['rainfall'].corr(df['Risk_Probability']),
        'Temperature': df['temperature'].corr(df['Risk_Probability']),
        'Humidity': df['humidity'].corr(df['Risk_Probability']),
        'pH': df['ph'].corr(df['Risk_Probability'])
    }
    
    print("🔍 Correlation Analysis:")
    for factor, corr in correlations.items():
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"   • {factor}: {corr:.3f} ({strength} {direction} correlation)")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Risk probability range: {df['Risk_Probability'].min():.3f} - {df['Risk_Probability'].max():.3f}")
    print(f"   • Average risk: {df['Risk_Probability'].mean():.3f}")
    
    print(f"\n🌡️ Environmental Factor Ranges:")
    print(f"   • Temperature: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
    print(f"   • Humidity: {df['humidity'].min():.1f}% - {df['humidity'].max():.1f}%")
    print(f"   • Rainfall: {df['rainfall'].min():.1f}mm - {df['rainfall'].max():.1f}mm")
    print(f"   • pH: {df['ph'].min():.1f} - {df['ph'].max():.1f}")

def main():
    """Main function to generate all scatter plot analyses"""
    print("🚀 Starting scatter plot analysis...")
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Create individual scatter plots
        print("\n📊 Creating individual scatter plots...")
        create_risk_vs_rainfall_plot(df)
        create_risk_vs_temperature_plot(df)
        create_risk_vs_humidity_plot(df)
        
        # Create multi-factor analysis
        print("\n🔍 Creating multi-factor analysis...")
        create_multi_factor_plot(df)
        create_correlation_heatmap(df)
        
        # Generate insights
        generate_insights(df)
        
        print("\n🎉 Scatter plot analysis completed!")
        print("📁 Files created:")
        print("   • risk_vs_rainfall.png")
        print("   • risk_vs_temperature.png")
        print("   • risk_vs_humidity.png")
        print("   • multi_factor_analysis.png")
        print("   • correlation_heatmap.png")
        
        print("\n📚 Research Paper Benefits:")
        print("   ✅ Visual correlation analysis")
        print("   ✅ Statistical relationship insights")
        print("   ✅ Environmental factor impact assessment")
        print("   ✅ Professional data visualization")
        print("   ✅ R² values for trend strength")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
