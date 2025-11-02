"""
Geographical Maps and Animated Time-Series Visualization for Agricultural Research
Creates risk level maps and animated disease/pest spread visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

print("🗺️ GEOGRAPHICAL VISUALIZATION GENERATOR")
print("="*60)

class AgriculturalMapVisualizer:
    """Class for creating geographical and temporal agricultural visualizations"""
    
    def __init__(self):
        self.risk_data = None
        self.yield_data = None
        self.temporal_data = None
        
    def generate_sample_geographical_data(self):
        """Generate sample geographical data for demonstration"""
        print("📊 Generating sample geographical data...")
        
        # Sample Indian states with agricultural data
        states = [
            'Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar', 'West Bengal',
            'Maharashtra', 'Gujarat', 'Rajasthan', 'Madhya Pradesh', 'Karnataka',
            'Tamil Nadu', 'Andhra Pradesh', 'Telangana', 'Kerala', 'Odisha',
            'Assam', 'Jharkhand', 'Chhattisgarh', 'Himachal Pradesh', 'Uttarakhand'
        ]
        
        # Generate risk levels (Low, Medium, High)
        risk_levels = np.random.choice(['Low', 'Medium', 'High'], size=len(states), p=[0.4, 0.4, 0.2])
        
        # Generate yield data (tonnes per hectare)
        yield_data = np.random.normal(3.5, 1.2, len(states))
        yield_data = np.clip(yield_data, 1.0, 6.0)  # Realistic range
        
        # Generate coordinates (approximate for Indian states)
        coordinates = {
            'Punjab': [30.8, 75.8], 'Haryana': [29.0, 76.1], 'Uttar Pradesh': [26.8, 80.9],
            'Bihar': [25.6, 85.1], 'West Bengal': [22.6, 88.4], 'Maharashtra': [19.8, 75.7],
            'Gujarat': [23.0, 72.6], 'Rajasthan': [27.0, 73.0], 'Madhya Pradesh': [22.9, 78.7],
            'Karnataka': [15.3, 76.5], 'Tamil Nadu': [11.1, 78.7], 'Andhra Pradesh': [15.8, 79.7],
            'Telangana': [18.1, 79.0], 'Kerala': [10.9, 76.2], 'Odisha': [20.3, 85.8],
            'Assam': [26.2, 92.9], 'Jharkhand': [23.6, 85.3], 'Chhattisgarh': [21.3, 81.9],
            'Himachal Pradesh': [31.1, 77.2], 'Uttarakhand': [30.1, 79.0]
        }
        
        # Create DataFrame
        self.risk_data = pd.DataFrame({
            'State': states,
            'Risk_Level': risk_levels,
            'Yield_Tonnes_Per_Hectare': yield_data,
            'Latitude': [coordinates[state][0] for state in states],
            'Longitude': [coordinates[state][1] for state in states]
        })
        
        print(f"✅ Generated data for {len(states)} states")
        return self.risk_data
    
    def generate_temporal_data(self):
        """Generate temporal data for animated visualization"""
        print("📅 Generating temporal data for animation...")
        
        # Generate 12 months of data
        months = pd.date_range('2024-01-01', periods=12, freq='M')
        
        # Sample crops
        crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton']
        
        temporal_data = []
        
        for month in months:
            for crop in crops:
                # Simulate disease/pest spread over time
                base_risk = np.random.uniform(0.1, 0.9)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month.month / 12)
                risk_score = base_risk * seasonal_factor
                
                # Simulate different regions
                for i, state in enumerate(['North', 'South', 'East', 'West', 'Central']):
                    regional_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 5)
                    final_risk = min(1.0, risk_score * regional_factor)
                    
                    temporal_data.append({
                        'Date': month,
                        'Crop': crop,
                        'Region': state,
                        'Risk_Score': final_risk,
                        'Disease_Spread': np.random.poisson(final_risk * 10),
                        'Pest_Count': np.random.poisson(final_risk * 15)
                    })
        
        self.temporal_data = pd.DataFrame(temporal_data)
        print(f"✅ Generated temporal data: {len(self.temporal_data)} records")
        return self.temporal_data
    
    def create_risk_level_map(self, save_path='risk_level_map.html'):
        """Create interactive risk level map using Plotly"""
        print("🗺️ Creating risk level map...")
        
        if self.risk_data is None:
            self.generate_sample_geographical_data()
        
        # Color mapping for risk levels
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        self.risk_data['Color'] = self.risk_data['Risk_Level'].map(color_map)
        
        # Create interactive map
        fig = px.scatter_mapbox(
            self.risk_data,
            lat='Latitude',
            lon='Longitude',
            color='Risk_Level',
            size='Yield_Tonnes_Per_Hectare',
            hover_name='State',
            hover_data={'Yield_Tonnes_Per_Hectare': ':.2f', 'Risk_Level': True},
            color_discrete_map=color_map,
            title='Agricultural Risk Levels and Yield by State',
            mapbox_style='open-street-map',
            zoom=5,
            center={'lat': 23.5, 'lon': 78.0}  # Center on India
        )
        
        fig.update_layout(
            title_font_size=16,
            title_x=0.5,
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        # Save as HTML
        fig.write_html(save_path)
        print(f"✅ Risk level map saved to {save_path}")
        
        return fig
    
    def create_yield_heatmap(self, save_path='yield_heatmap.png'):
        """Create yield heatmap visualization"""
        print("🌾 Creating yield heatmap...")
        
        if self.risk_data is None:
            self.generate_sample_geographical_data()
        
        # Create pivot table for heatmap
        pivot_data = self.risk_data.pivot_table(
            values='Yield_Tonnes_Per_Hectare',
            index='Risk_Level',
            columns='State',
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Yield (Tonnes/Hectare)'})
        
        plt.title('Agricultural Yield Heatmap by State and Risk Level', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('States', fontsize=12, fontweight='bold')
        plt.ylabel('Risk Level', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Yield heatmap saved to {save_path}")
        plt.show()
    
    def create_animated_timeseries(self, save_path='animated_timeseries.gif'):
        """Create animated time-series showing disease/pest spread"""
        print("🎬 Creating animated time-series...")
        
        if self.temporal_data is None:
            self.generate_temporal_data()
        
        # Prepare data for animation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Get data for current month
            current_data = self.temporal_data[
                self.temporal_data['Date'] == self.temporal_data['Date'].unique()[frame]
            ]
            
            # Plot 1: Disease spread by region
            regions = current_data['Region'].unique()
            disease_counts = [current_data[current_data['Region'] == region]['Disease_Spread'].sum() 
                            for region in regions]
            
            bars1 = ax1.bar(regions, disease_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax1.set_title(f'Disease Spread by Region - {current_data["Date"].iloc[0].strftime("%B %Y")}', 
                         fontweight='bold')
            ax1.set_ylabel('Disease Cases', fontweight='bold')
            ax1.set_ylim(0, max(disease_counts) * 1.1)
            
            # Add values on bars
            for bar, count in zip(bars1, disease_counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Pest count by crop
            crops = current_data['Crop'].unique()
            pest_counts = [current_data[current_data['Crop'] == crop]['Pest_Count'].sum() 
                          for crop in crops]
            
            bars2 = ax2.bar(crops, pest_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            ax2.set_title(f'Pest Count by Crop - {current_data["Date"].iloc[0].strftime("%B %Y")}', 
                         fontweight='bold')
            ax2.set_ylabel('Pest Count', fontweight='bold')
            ax2.set_ylim(0, max(pest_counts) * 1.1)
            
            # Add values on bars
            for bar, count in zip(bars2, pest_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(self.temporal_data['Date'].unique()), 
                           interval=1000, repeat=True)
        
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=1)
        print(f"✅ Animated time-series saved to {save_path}")
        
        return anim
    
    def create_risk_trend_analysis(self, save_path='risk_trend_analysis.png'):
        """Create risk trend analysis over time"""
        print("📈 Creating risk trend analysis...")
        
        if self.temporal_data is None:
            self.generate_temporal_data()
        
        # Calculate monthly average risk by region
        monthly_risk = self.temporal_data.groupby(['Date', 'Region'])['Risk_Score'].mean().reset_index()
        
        # Create line plot
        plt.figure(figsize=(14, 8))
        
        for region in monthly_risk['Region'].unique():
            region_data = monthly_risk[monthly_risk['Region'] == region]
            plt.plot(region_data['Date'], region_data['Risk_Score'], 
                    marker='o', linewidth=2, label=region, markersize=6)
        
        plt.title('Agricultural Risk Trends by Region Over Time', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Average Risk Score', fontsize=12, fontweight='bold')
        plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Risk trend analysis saved to {save_path}")
        plt.show()

def main():
    """Main function to generate all geographical visualizations"""
    print("🚀 Starting geographical visualization generation...")
    
    # Initialize visualizer
    visualizer = AgriculturalMapVisualizer()
    
    try:
        # Generate sample data
        visualizer.generate_sample_geographical_data()
        visualizer.generate_temporal_data()
        
        # Create visualizations
        print("\n🗺️ Creating geographical maps...")
        visualizer.create_risk_level_map()
        visualizer.create_yield_heatmap()
        
        print("\n🎬 Creating animated visualizations...")
        visualizer.create_animated_timeseries()
        visualizer.create_risk_trend_analysis()
        
        print("\n🎉 All geographical visualizations completed!")
        print("📁 Files created:")
        print("   • risk_level_map.html - Interactive risk map")
        print("   • yield_heatmap.png - Yield heatmap")
        print("   • animated_timeseries.gif - Animated disease/pest spread")
        print("   • risk_trend_analysis.png - Risk trend analysis")
        
        print("\n📚 Research Paper Benefits:")
        print("   ✅ Visual impact and engagement")
        print("   ✅ Spatial context for agricultural planning")
        print("   ✅ Temporal analysis of disease/pest spread")
        print("   ✅ Policy-relevant insights")
        print("   ✅ Professional presentation quality")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
