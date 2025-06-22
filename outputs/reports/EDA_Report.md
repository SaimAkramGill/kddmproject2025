
# Superhero Dataset - Exploratory Data Analysis Report
Generated on: 2025-06-22 15:36:19

## Executive Summary
This report presents a comprehensive exploratory data analysis of the superhero dataset containing 1041 characters with 24 features.

## Dataset Overview
- **Total Records**: 1,041
- **Total Features**: 24
- **Numerical Features**: 12
- **Categorical Features**: 11
- **Memory Usage**: 0.19 MB
- **Duplicate Records**: 49

## Target Variable Analysis

### Winning Probability Statistics
- **Mean**: 0.8102
- **Median**: 0.6000
- **Range**: [0.2200, 15.2000]
- **Standard Deviation**: 1.5649

### Binary Classification Distribution

- **Winners** (>0.5): 766 (73.6%)
- **Losers** (≤0.5): 275 (26.4%)

## Data Quality Assessment

### Missing Values
Total missing values: 1,701

#### Columns with >5% Missing Values:
- **secret_code**: 388 (37.3%)
- **eye_color**: 337 (32.4%)
- **speed**: 283 (27.2%)
- **gender**: 176 (16.9%)
- **hair_color**: 101 (9.7%)
- **training_time**: 96 (9.2%)
- **intelligence**: 76 (7.3%)
- **weight**: 72 (6.9%)
- **power_level**: 64 (6.1%)

## Feature Correlation Analysis

### Top 5 Features Most Correlated with Win Probability:
1. **ranking**: 0.0690 (negative)
1. **battle_iq**: 0.0470 (positive)
1. **intelligence**: 0.0430 (positive)
1. **speed**: 0.0425 (positive)
1. **height**: 0.0279 (negative)

## Key Insights

### Data Distribution Patterns
- The dataset shows imbalanced class distribution
- Categorical features appear consistent

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


