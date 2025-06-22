"""
Superhero EDA (Exploratory Data Analysis) Module
Comprehensive data exploration and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class SuperheroEDA:
    """
    Comprehensive Exploratory Data Analysis for Superhero Dataset
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.numerical_features = []
        self.categorical_features = []
        self.missing_summary = None
        self.correlation_matrix = None
        
        # Set visualization style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load and perform initial data inspection"""
        print(" Loading superhero dataset...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f" Dataset loaded successfully!")
            print(f" Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            
            # Identify feature types
            self.numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
            
            print(f" Numerical features: {len(self.numerical_features)}")
            print(f" Categorical features: {len(self.categorical_features)}")
            
            return self.data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\n BASIC STATISTICAL ANALYSIS")
        print("-" * 50)
        
        basic_stats = {
            'dataset_shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage().sum() / 1024**2,
            'duplicate_rows': self.data.duplicated().sum(),
            'total_missing_values': self.data.isnull().sum().sum()
        }
        
        print(f"Dataset dimensions: {basic_stats['dataset_shape']}")
        print(f"Memory usage: {basic_stats['memory_usage_mb']:.2f} MB")
        print(f"Duplicate rows: {basic_stats['duplicate_rows']}")
        print(f"Total missing values: {basic_stats['total_missing_values']}")
        
        # Target variable analysis
        if 'win_prob' in self.data.columns:
            win_prob_stats = self.data['win_prob'].describe()
            print(f"\n Target Variable (win_prob) Statistics:")
            print(win_prob_stats)
            
            # Create binary target
            self.data['win_binary'] = (self.data['win_prob'] > 0.5).astype(int)
            class_distribution = self.data['win_binary'].value_counts()
            print(f"\n Binary Classification Distribution:")
            print(f"Winners (>0.5): {class_distribution.get(1, 0)} ({class_distribution.get(1, 0)/len(self.data)*100:.1f}%)")
            print(f"Losers (≤0.5): {class_distribution.get(0, 0)} ({class_distribution.get(0, 0)/len(self.data)*100:.1f}%)")
            
            basic_stats['target_stats'] = win_prob_stats.to_dict()
            basic_stats['class_distribution'] = class_distribution.to_dict()
        
        return basic_stats
    
    def data_quality_analysis(self):
        """Analyze data quality issues"""
        print("\n DATA QUALITY ANALYSIS")
        print("-" * 50)
        
        # Missing value analysis
        missing_data = pd.DataFrame({
            'Column': self.data.columns,
            'Missing_Count': self.data.isnull().sum().values,
            'Missing_Percentage': (self.data.isnull().sum().values / len(self.data)) * 100,
            'Data_Type': [str(dtype) for dtype in self.data.dtypes]
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.missing_summary = missing_data
        
        print("Missing Values Summary:")
        print(missing_data.head(10).to_string(index=False))
        
        # Identify high missing value columns
        high_missing = missing_data[missing_data['Missing_Percentage'] > 20]
        if not high_missing.empty:
            print(f"\n Columns with >20% missing values:")
            for _, row in high_missing.iterrows():
                print(f"  • {row['Column']}: {row['Missing_Percentage']:.1f}%")
        
        # Check for inconsistent categorical values
        quality_issues = {}
        
        for col in self.categorical_features:
            if col in self.data.columns:
                unique_values = self.data[col].value_counts()
                
                # Check for potential spelling variations
                if col == 'role':
                    similar_values = []
                    for val in unique_values.index:
                        if pd.notna(val):
                            val_str = str(val).lower()
                            if 'hero' in val_str or 'villain' in val_str:
                                similar_values.append(val)
                    
                    if len(similar_values) > 2:
                        quality_issues[col] = similar_values
                        print(f"\n🚨 Inconsistent spellings in '{col}':")
                        for val in similar_values:
                            print(f"  • {val}: {unique_values[val]} occurrences")
        
        return {
            'missing_summary': missing_data,
            'quality_issues': quality_issues
        }
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n CORRELATION ANALYSIS")
        print("-" * 50)
        
        # Calculate correlation matrix for numerical features
        numerical_data = self.data[self.numerical_features].select_dtypes(include=[np.number])
        self.correlation_matrix = numerical_data.corr()
        
        # Find features most correlated with target
        if 'win_prob' in self.correlation_matrix.columns:
            target_correlations = self.correlation_matrix['win_prob'].abs().sort_values(ascending=False)
            
            print("Features most correlated with win_prob:")
            for feature, corr in target_correlations.head(10).items():
                if feature != 'win_prob':
                    direction = "positive" if self.correlation_matrix.loc[feature, 'win_prob'] > 0 else "negative"
                    print(f"  • {feature}: {corr:.4f} ({direction})")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if high_corr_pairs:
            print(f"\n🔄 Highly correlated feature pairs (>0.7):")
            for pair in high_corr_pairs:
                print(f"  • {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'target_correlations': target_correlations if 'win_prob' in self.correlation_matrix.columns else None,
            'high_corr_pairs': high_corr_pairs
        }
    
    def feature_analysis(self):
        """Detailed analysis of individual features"""
        print("\n INDIVIDUAL FEATURE ANALYSIS")
        print("-" * 50)
        
        feature_insights = {}
        
        # Analyze numerical features
        for feature in self.numerical_features[:10]:  # Limit to first 10 for brevity
            if feature in self.data.columns:
                feature_data = self.data[feature].dropna()
                
                if len(feature_data) > 0:
                    # Basic statistics
                    stats = feature_data.describe()
                    
                    # Outlier detection using IQR
                    Q1 = feature_data.quantile(0.25)
                    Q3 = feature_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
                    
                    feature_insights[feature] = {
                        'stats': stats.to_dict(),
                        'outlier_count': len(outliers),
                        'outlier_percentage': len(outliers) / len(feature_data) * 100,
                        'skewness': feature_data.skew(),
                        'kurtosis': feature_data.kurtosis()
                    }
                    
                    print(f"\n{feature}:")
                    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
                    print(f"  Mean: {stats['mean']:.2f}, Median: {stats['50%']:.2f}")
                    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(feature_data)*100:.1f}%)")
                    if abs(feature_data.skew()) > 1:
                        skew_direction = "right" if feature_data.skew() > 0 else "left"
                        print(f"  Distribution: Heavily skewed {skew_direction}")
        
        # Analyze categorical features
        for feature in self.categorical_features[:5]:  # Limit to first 5
            if feature in self.data.columns:
                value_counts = self.data[feature].value_counts()
                
                feature_insights[feature] = {
                    'unique_values': len(value_counts),
                    'most_common': value_counts.head().to_dict(),
                    'least_common': value_counts.tail().to_dict()
                }
                
                print(f"\n{feature}:")
                print(f"  Unique values: {len(value_counts)}")
                print(f"  Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
                if len(value_counts) > 1:
                    print(f"  Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
        
        return feature_insights
    
    def create_overview_plots(self):
        """Create overview visualizations"""
        print("\n Creating overview plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Superhero Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        if 'win_prob' in self.data.columns:
            axes[0,0].hist(self.data['win_prob'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('Distribution of Winning Probability')
            axes[0,0].set_xlabel('Winning Probability')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Missing values heatmap
        missing_data = self.data.isnull()
        if missing_data.any().any():
            missing_cols = missing_data.sum()[missing_data.sum() > 0].index
            if len(missing_cols) > 0:
                missing_subset = missing_data[missing_cols].head(100)  # First 100 rows
                sns.heatmap(missing_subset.T, cbar=True, yticklabels=True, xticklabels=False, 
                           cmap='viridis', ax=axes[0,1])
                axes[0,1].set_title('Missing Values Pattern (First 100 rows)')
                axes[0,1].set_xlabel('Samples')
        
        # 3. Class distribution (binary)
        if 'win_binary' in self.data.columns:
            class_counts = self.data['win_binary'].value_counts()
            axes[1,0].pie(class_counts.values, labels=['Losers', 'Winners'], autopct='%1.1f%%', 
                         colors=['lightcoral', 'lightgreen'])
            axes[1,0].set_title('Binary Classification Distribution')
        
        # 4. Feature types distribution
        feature_types = {
            'Numerical': len(self.numerical_features),
            'Categorical': len(self.categorical_features)
        }
        axes[1,1].bar(feature_types.keys(), feature_types.values(), color=['lightblue', 'lightyellow'])
        axes[1,1].set_title('Feature Types Distribution')
        axes[1,1].set_ylabel('Number of Features')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/01_dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_plots(self):
        """Create correlation visualizations"""
        print(" Creating correlation plots...")
        
        if self.correlation_matrix is not None:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # 1. Full correlation heatmap
            mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
            sns.heatmap(self.correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f', ax=axes[0])
            axes[0].set_title('Feature Correlation Matrix')
            
            # 2. Target correlations bar plot
            if 'win_prob' in self.correlation_matrix.columns:
                target_corr = self.correlation_matrix['win_prob'].drop('win_prob').abs().sort_values(ascending=True)
                target_corr.tail(10).plot(kind='barh', ax=axes[1], color='orange')
                axes[1].set_title('Top 10 Features Correlated with Winning Probability')
                axes[1].set_xlabel('Absolute Correlation')
            
            plt.tight_layout()
            plt.savefig('outputs/plots/02_correlation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_distribution_plots(self):
        """Create feature distribution plots"""
        print(" Creating distribution plots...")
        
        # Numerical features distributions
        numerical_cols = [col for col in self.numerical_features if col in self.data.columns][:8]
        
        if numerical_cols:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                data_clean = self.data[col].dropna()
                if len(data_clean) > 0:
                    axes[i].hist(data_clean, bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('outputs/plots/03_feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_target_analysis_plots(self):
        """Create target variable analysis plots"""
        print(" Creating target analysis plots...")
        
        if 'win_prob' not in self.data.columns:
            print(" No win_prob column found, skipping target analysis plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold')
        
        # 1. Win probability by role
        if 'role' in self.data.columns:
            try:
                role_win_prob = self.data.groupby('role')['win_prob'].mean().sort_values(ascending=False)
                if len(role_win_prob) > 0:
                    role_win_prob.plot(kind='bar', ax=axes[0,0], color='lightblue')
                    axes[0,0].set_title('Average Win Probability by Role')
                    axes[0,0].set_ylabel('Average Win Probability')
                    axes[0,0].tick_params(axis='x', rotation=45)
                else:
                    axes[0,0].text(0.5, 0.5, 'No role data available', ha='center', va='center', transform=axes[0,0].transAxes)
            except Exception as e:
                axes[0,0].text(0.5, 0.5, f'Role analysis error:\n{str(e)[:30]}...', ha='center', va='center', transform=axes[0,0].transAxes)
        
        # 2. Win probability vs top correlated feature
        if hasattr(self, 'correlation_matrix') and 'win_prob' in self.correlation_matrix.columns:
            try:
                top_corr_feature = self.correlation_matrix['win_prob'].abs().drop('win_prob').idxmax()
                if top_corr_feature in self.data.columns:
                    # Clean data for scatter plot
                    clean_data = self.data[[top_corr_feature, 'win_prob']].dropna()
                    if len(clean_data) > 0:
                        axes[0,1].scatter(clean_data[top_corr_feature], clean_data['win_prob'], alpha=0.6)
                        axes[0,1].set_xlabel(top_corr_feature)
                        axes[0,1].set_ylabel('Win Probability')
                        axes[0,1].set_title(f'Win Probability vs {top_corr_feature}')
                        axes[0,1].grid(True, alpha=0.3)
                    else:
                        axes[0,1].text(0.5, 0.5, 'No clean data for scatter', ha='center', va='center', transform=axes[0,1].transAxes)
                else:
                    axes[0,1].text(0.5, 0.5, 'Top correlated feature not found', ha='center', va='center', transform=axes[0,1].transAxes)
            except Exception as e:
                axes[0,1].text(0.5, 0.5, f'Correlation plot error:\n{str(e)[:30]}...', ha='center', va='center', transform=axes[0,1].transAxes)
        
        # 3. Box plot of win probability by binary outcome
        if 'win_binary' in self.data.columns:
            try:
                win_data = [self.data[self.data['win_binary']==0]['win_prob'].dropna(),
                           self.data[self.data['win_binary']==1]['win_prob'].dropna()]
                # Filter out empty arrays
                win_data = [data for data in win_data if len(data) > 0]
                if len(win_data) >= 2:
                    axes[1,0].boxplot(win_data, labels=['Losers', 'Winners'])
                    axes[1,0].set_title('Win Probability Distribution by Outcome')
                    axes[1,0].set_ylabel('Win Probability')
                else:
                    axes[1,0].text(0.5, 0.5, 'Insufficient data for boxplot', ha='center', va='center', transform=axes[1,0].transAxes)
            except Exception as e:
                axes[1,0].text(0.5, 0.5, f'Boxplot error:\n{str(e)[:30]}...', ha='center', va='center', transform=axes[1,0].transAxes)
        
        # 4. Win rate by power level bins
        if 'power_level' in self.data.columns:
            try:
                power_data = self.data[['power_level', 'win_binary']].dropna()
                if len(power_data) > 0:
                    power_data['power_bins'] = pd.cut(power_data['power_level'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    power_win_rate = power_data.groupby('power_bins')['win_binary'].mean()
                    if len(power_win_rate) > 0:
                        power_win_rate.plot(kind='bar', ax=axes[1,1], color='lightgreen')
                        axes[1,1].set_title('Win Rate by Power Level')
                        axes[1,1].set_ylabel('Win Rate')
                        axes[1,1].tick_params(axis='x', rotation=45)
                    else:
                        axes[1,1].text(0.5, 0.5, 'No power level data', ha='center', va='center', transform=axes[1,1].transAxes)
                else:
                    axes[1,1].text(0.5, 0.5, 'No clean power/win data', ha='center', va='center', transform=axes[1,1].transAxes)
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f'Power analysis error:\n{str(e)[:30]}...', ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/04_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        print(" Creating interactive plots...")
        
        # Interactive correlation heatmap
        if self.correlation_matrix is not None:
            fig = px.imshow(self.correlation_matrix, 
                           color_continuous_scale='RdBu',
                           title='Interactive Feature Correlation Matrix')
            fig.write_html('outputs/plots/interactive_correlation.html')
        
        # Interactive scatter plot
        if 'win_prob' in self.data.columns and 'power_level' in self.data.columns:
            # Clean data for plotting - remove NaN values
            plot_data = self.data[['power_level', 'win_prob']].copy()
            
            # Add additional columns if they exist
            if 'role' in self.data.columns:
                plot_data['role'] = self.data['role']
            
            if 'speed' in self.data.columns:
                plot_data['speed'] = self.data['speed']
            
            if 'name' in self.data.columns:
                plot_data['name'] = self.data['name']
            
            # Remove rows with NaN in essential columns
            plot_data = plot_data.dropna(subset=['power_level', 'win_prob'])
            
            if len(plot_data) > 0:
                # Create size column, handling NaN values
                if 'speed' in plot_data.columns:
                    # Fill NaN speed values with median
                    speed_median = plot_data['speed'].median()
                    plot_data['speed_clean'] = plot_data['speed'].fillna(speed_median)
                    # Normalize speed for size (between 5 and 50)
                    speed_min = plot_data['speed_clean'].min()
                    speed_max = plot_data['speed_clean'].max()
                    if speed_max > speed_min:
                        plot_data['size_normalized'] = 5 + 45 * (plot_data['speed_clean'] - speed_min) / (speed_max - speed_min)
                    else:
                        plot_data['size_normalized'] = 20  # Default size
                else:
                    plot_data['size_normalized'] = 20  # Default size if no speed column
                
                # Create the plot
                fig = px.scatter(
                    plot_data, 
                    x='power_level', 
                    y='win_prob',
                    color='role' if 'role' in plot_data.columns else None,
                    size='size_normalized',
                    hover_data=['name'] if 'name' in plot_data.columns else None,
                    title='Interactive Win Probability Analysis',
                    labels={
                        'power_level': 'Power Level',
                        'win_prob': 'Win Probability',
                        'size_normalized': 'Speed (normalized)'
                    }
                )
                
                fig.write_html('outputs/plots/interactive_scatter.html')
            else:
                print(" Not enough clean data for interactive scatter plot")
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report"""
        print(" Generating EDA report...")
        
        report_content = f"""
# Superhero Dataset - Exploratory Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents a comprehensive exploratory data analysis of the superhero dataset containing {self.data.shape[0]} characters with {self.data.shape[1]} features.

## Dataset Overview
- **Total Records**: {self.data.shape[0]:,}
- **Total Features**: {self.data.shape[1]}
- **Numerical Features**: {len(self.numerical_features)}
- **Categorical Features**: {len(self.categorical_features)}
- **Memory Usage**: {self.data.memory_usage().sum() / 1024**2:.2f} MB
- **Duplicate Records**: {self.data.duplicated().sum()}

## Target Variable Analysis
"""
        
        if 'win_prob' in self.data.columns:
            win_stats = self.data['win_prob'].describe()
            class_dist = self.data['win_binary'].value_counts() if 'win_binary' in self.data.columns else None
            
            report_content += f"""
### Winning Probability Statistics
- **Mean**: {win_stats['mean']:.4f}
- **Median**: {win_stats['50%']:.4f}
- **Range**: [{win_stats['min']:.4f}, {win_stats['max']:.4f}]
- **Standard Deviation**: {win_stats['std']:.4f}

### Binary Classification Distribution
"""
            if class_dist is not None:
                report_content += f"""
- **Winners** (>0.5): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(self.data)*100:.1f}%)
- **Losers** (≤0.5): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(self.data)*100:.1f}%)
"""
        
        # Missing values analysis
        if self.missing_summary is not None:
            high_missing = self.missing_summary[self.missing_summary['Missing_Percentage'] > 5]
            report_content += f"""
## Data Quality Assessment

### Missing Values
Total missing values: {self.data.isnull().sum().sum():,}

#### Columns with >5% Missing Values:
"""
            for _, row in high_missing.iterrows():
                report_content += f"- **{row['Column']}**: {row['Missing_Count']} ({row['Missing_Percentage']:.1f}%)\n"
        
        # Correlation insights
        if hasattr(self, 'correlation_matrix') and 'win_prob' in self.correlation_matrix.columns:
            target_corr = self.correlation_matrix['win_prob'].abs().sort_values(ascending=False)
            report_content += f"""
## Feature Correlation Analysis

### Top 5 Features Most Correlated with Win Probability:
"""
            for feature, corr in target_corr.head(6).items():  # Top 5 + target itself
                if feature != 'win_prob':
                    direction = "positive" if self.correlation_matrix.loc[feature, 'win_prob'] > 0 else "negative"
                    report_content += f"1. **{feature}**: {corr:.4f} ({direction})\n"
        
        # Key insights
        report_content += f"""
## Key Insights

### Data Distribution Patterns
- The dataset shows {'balanced' if abs(self.data['win_binary'].mean() - 0.5) < 0.1 else 'imbalanced'} class distribution
- {'Multiple spelling variations detected in categorical features' if any('role' in issues for issues in getattr(self, 'quality_issues', {}).keys()) else 'Categorical features appear consistent'}

### Feature Characteristics
- Numerical features show varying scales requiring normalization
- Several features exhibit right-skewed distributions
- Outliers detected in multiple numerical features

### Modeling Recommendations
1. **Data Preprocessing**: Handle missing values through imputation
2. **Feature Engineering**: Standardize categorical spellings, normalize numerical features
3. **Class Imbalance**: Consider balanced sampling or weighted algorithms
4. **Feature Selection**: Focus on top correlated features for initial modeling

## Files Generated
- `01_dataset_overview.png`: Basic dataset visualizations
- `02_correlation_analysis.png`: Feature correlation heatmaps
- `03_feature_distributions.png`: Individual feature distributions
- `04_target_analysis.png`: Target variable analysis
- `interactive_correlation.html`: Interactive correlation matrix
- `interactive_scatter.html`: Interactive scatter plots

---
*Report generated by Superhero EDA Module*
"""
        
        # Save report
        with open('outputs/reports/EDA_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Also save as text for easy reading
        with open('outputs/reports/EDA_Summary.txt', 'w', encoding='utf-8') as f:
            f.write(report_content.replace('#', '').replace('*', ''))
        
        print(" EDA report saved to outputs/reports/")
    
    def get_summary_stats(self):
        """Return summary statistics for other modules"""
        return {
            'dataset_shape': self.data.shape,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'missing_summary': self.missing_summary,
            'correlation_matrix': self.correlation_matrix,
            'target_stats': self.data['win_prob'].describe().to_dict() if 'win_prob' in self.data.columns else None,
            'class_distribution': self.data['win_binary'].value_counts().to_dict() if 'win_binary' in self.data.columns else None
        }

if __name__ == "__main__":
    # Test the EDA module
    print(" Testing SuperheroEDA module...")
    
    eda = SuperheroEDA('data.csv')
    eda.load_data()
    eda.basic_analysis()
    eda.correlation_analysis()
    eda.feature_analysis()
    eda.data_quality_analysis()
    
    # Create all visualizations
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    eda.create_overview_plots()
    eda.create_correlation_plots()
    eda.create_distribution_plots()
    eda.create_target_analysis_plots()
    eda.create_interactive_plots()
    eda.generate_eda_report()
    
    print(" EDA module test completed!")