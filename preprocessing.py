"""
Data Preprocessing Module for Superhero Dataset
Handles cleaning, encoding, imputation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for superhero dataset
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.original_data = data.copy()
        self.label_encoders = {}
        self.scaler = None
        self.imputers = {}
        self.feature_names = []
        self.preprocessing_log = []
        
    def log_step(self, step_name, details):
        """Log preprocessing steps"""
        log_entry = {
            'step': step_name,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'details': details
        }
        self.preprocessing_log.append(log_entry)
        print(f" {step_name}: {details}")
    
    def clean_target_variable(self):
        """Clean and prepare target variable"""
        print("\n CLEANING TARGET VARIABLE")
        print("-" * 40)
        
        initial_rows = len(self.data)
        
        # Remove rows with missing target
        self.data = self.data.dropna(subset=['win_prob'])
        removed_rows = initial_rows - len(self.data)
        
        if removed_rows > 0:
            self.log_step("Target Cleaning", f"Removed {removed_rows} rows with missing win_prob")
        
        # Handle extreme values
        initial_min = self.data['win_prob'].min()
        initial_max = self.data['win_prob'].max()
        
        # Cap extreme values
        self.data['win_prob'] = np.clip(self.data['win_prob'], 0, 1)
        
        capped_values = ((self.data['win_prob'] == 0) | (self.data['win_prob'] == 1)).sum()
        self.log_step("Value Capping", f"Capped {capped_values} extreme values to [0,1] range")
        
        # Create binary target
        self.data['win_binary'] = (self.data['win_prob'] > 0.5).astype(int)
        class_distribution = self.data['win_binary'].value_counts()
        
        self.log_step("Binary Target", f"Created binary target: {class_distribution[1]} winners, {class_distribution[0]} losers")
        
        return self.data
    
    def standardize_categorical_values(self):
        """Standardize inconsistent categorical values"""
        print("\n STANDARDIZING CATEGORICAL VALUES")
        print("-" * 40)
        
        # Standardize role spellings
        if 'role' in self.data.columns:
            role_mapping = {
                'H3ro': 'Hero',
                'HerO': 'Hero',
                'VIllain': 'Villain',
                'VillaIn': 'Villain'
            }
            
            initial_unique = self.data['role'].nunique()
            self.data['role'] = self.data['role'].map(role_mapping).fillna(self.data['role'])
            final_unique = self.data['role'].nunique()
            
            self.log_step("Role Standardization", f"Reduced role categories from {initial_unique} to {final_unique}")
        
        # Clean other categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'role':  # Already handled
                # Remove extra whitespace and standardize case
                original_unique = self.data[col].nunique()
                self.data[col] = self.data[col].astype(str).str.strip().str.title()
                
                # Replace 'Nan' string with actual NaN
                self.data[col] = self.data[col].replace(['Nan', 'None', 'Unknown'], np.nan)
                
                new_unique = self.data[col].nunique()
                if original_unique != new_unique:
                    self.log_step(f"{col} Cleaning", f"Reduced unique values from {original_unique} to {new_unique}")
        
        return self.data
    
    def handle_missing_values(self):
        """Comprehensive missing value handling"""
        print("\n HANDLING MISSING VALUES")
        print("-" * 40)
        
        # Analyze missing patterns
        missing_before = self.data.isnull().sum()
        high_missing_cols = missing_before[missing_before > len(self.data) * 0.5].index.tolist()
        
        if high_missing_cols:
            self.log_step("High Missing Columns", f"Dropping columns with >50% missing: {high_missing_cols}")
            self.data = self.data.drop(columns=high_missing_cols)
        
        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target columns from imputation
        numerical_cols = [col for col in numerical_cols if col not in ['win_prob', 'win_binary']]
        
        # Impute numerical features
        if numerical_cols:
            # Use median imputation for most features
            median_imputer = SimpleImputer(strategy='median')
            self.data[numerical_cols] = median_imputer.fit_transform(self.data[numerical_cols])
            self.imputers['numerical_median'] = median_imputer
            
            self.log_step("Numerical Imputation", f"Applied median imputation to {len(numerical_cols)} numerical features")
        
        # Impute categorical features
        if categorical_cols:
            mode_imputer = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = mode_imputer.fit_transform(self.data[categorical_cols])
            self.imputers['categorical_mode'] = mode_imputer
            
            self.log_step("Categorical Imputation", f"Applied mode imputation to {len(categorical_cols)} categorical features")
        
        # Verify no missing values remain
        remaining_missing = self.data.isnull().sum().sum()
        if remaining_missing == 0:
            self.log_step("Missing Value Check", "All missing values successfully handled")
        else:
            self.log_step("Warning", f"{remaining_missing} missing values still remain")
        
        return self.data
    
    def encode_categorical_features(self):
        """Encode categorical features for machine learning"""
        print("\n ENCODING CATEGORICAL FEATURES")
        print("-" * 40)
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col in self.data.columns:
                # Create label encoder
                le = LabelEncoder()
                
                # Handle any remaining NaN values
                self.data[col] = self.data[col].fillna('Unknown')
                
                # Fit and transform
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                self.label_encoders[col] = le
                
                unique_values = len(le.classes_)
                self.log_step(f"{col} Encoding", f"Label encoded {unique_values} unique values")
        
        # Drop original categorical columns
        self.data = self.data.drop(columns=categorical_cols)
        
        return self.data
    
    def feature_engineering(self):
        """Create new features from existing ones"""
        print("\n FEATURE ENGINEERING")
        print("-" * 40)
        
        new_features_count = 0
        
        # Power to weight ratio (if both exist)
        if 'power_level' in self.data.columns and 'weight' in self.data.columns:
            self.data['power_weight_ratio'] = self.data['power_level'] / (self.data['weight'] + 1)  # +1 to avoid division by zero
            new_features_count += 1
            self.log_step("Power-Weight Ratio", "Created power to weight ratio feature")
        
        # Speed to age ratio (experience-adjusted speed)
        if 'speed' in self.data.columns and 'age' in self.data.columns:
            self.data['speed_age_ratio'] = self.data['speed'] / (self.data['age'] + 1)
            new_features_count += 1
            self.log_step("Speed-Age Ratio", "Created experience-adjusted speed feature")
        
        # Intelligence to battle_iq ratio (tactical intelligence)
        if 'intelligence' in self.data.columns and 'battle_iq' in self.data.columns:
            self.data['tactical_intelligence'] = (self.data['intelligence'] + self.data['battle_iq']) / 2
            new_features_count += 1
            self.log_step("Tactical Intelligence", "Created combined intelligence feature")
        
        # BMI-like feature (if height and weight exist)
        if 'height' in self.data.columns and 'weight' in self.data.columns:
            # Superhero BMI = weight / (height/100)^2
            self.data['hero_bmi'] = self.data['weight'] / ((self.data['height'] / 100) ** 2 + 0.01)
            new_features_count += 1
            self.log_step("Hero BMI", "Created body mass index feature")
        
        # Training efficiency (training_time vs age)
        if 'training_time' in self.data.columns and 'age' in self.data.columns:
            self.data['training_efficiency'] = self.data['training_time'] / (self.data['age'] + 1)
            new_features_count += 1
            self.log_step("Training Efficiency", "Created training efficiency feature")
        
        # Ranking category (convert numerical ranking to categories)
        if 'ranking' in self.data.columns:
            self.data['ranking_category'] = pd.cut(self.data['ranking'], 
                                                 bins=5, 
                                                 labels=['Elite', 'High', 'Medium', 'Low', 'Rookie'])
            # Convert categories to numerical
            ranking_mapping = {'Elite': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Rookie': 1}
            self.data['ranking_category_num'] = self.data['ranking_category'].map(ranking_mapping)
            self.data = self.data.drop('ranking_category', axis=1)
            new_features_count += 1
            self.log_step("Ranking Categories", "Created categorical ranking feature")
        
        self.log_step("Feature Engineering", f"Created {new_features_count} new features")
        
        return self.data
    
    def handle_outliers(self, method='iqr', threshold=1.5):
        """Handle outliers in numerical features"""
        print(f"\n HANDLING OUTLIERS (method: {method.upper()})")
        print("-" * 40)
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target variables and engineered ratios that might naturally have extreme values
        exclude_cols = ['win_prob', 'win_binary'] + [col for col in numerical_cols if 'ratio' in col.lower()]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        outliers_handled = {}
        
        for col in numerical_cols:
            if col in self.data.columns:
                initial_outliers = 0
                
                if method == 'iqr':
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                    initial_outliers = outlier_mask.sum()
                    
                    if initial_outliers > 0:
                        # Cap outliers instead of removing them
                        self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                        self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                        
                        outliers_handled[col] = initial_outliers
                
                elif method == 'zscore':
                    z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                    outlier_mask = z_scores > threshold
                    initial_outliers = outlier_mask.sum()
                    
                    if initial_outliers > 0:
                        # Cap to mean ± threshold * std
                        mean_val = self.data[col].mean()
                        std_val = self.data[col].std()
                        lower_bound = mean_val - threshold * std_val
                        upper_bound = mean_val + threshold * std_val
                        
                        self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                        self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                        
                        outliers_handled[col] = initial_outliers
        
        total_outliers = sum(outliers_handled.values())
        if total_outliers > 0:
            self.log_step("Outlier Handling", f"Capped {total_outliers} outliers across {len(outliers_handled)} features")
        else:
            self.log_step("Outlier Check", "No significant outliers detected")
        
        return self.data
    
    def scale_features(self, method='standard'):
        """Scale numerical features"""
        print(f"\n SCALING FEATURES (method: {method})")
        print("-" * 40)
        
        # Get numerical columns (excluding target)
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in ['win_prob', 'win_binary']]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
        
        if feature_cols:
            # Fit scaler on features only
            scaled_features = self.scaler.fit_transform(self.data[feature_cols])
            
            # Create scaled dataframe
            scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=self.data.index)
            
            # Replace original features with scaled versions
            for col in feature_cols:
                self.data[col] = scaled_df[col]
            
            self.log_step("Feature Scaling", f"Applied {method} scaling to {len(feature_cols)} features")
        
        return self.data
    
    def prepare_model_features(self):
        """Prepare final feature set for modeling"""
        print("\n PREPARING MODEL FEATURES")
        print("-" * 40)
        
        # Define feature columns (exclude target variables)
        all_columns = self.data.columns.tolist()
        target_columns = ['win_prob', 'win_binary']
        
        # Also exclude non-predictive columns if they exist
        exclude_columns = target_columns + ['name'] if 'name' in all_columns else target_columns
        
        self.feature_names = [col for col in all_columns if col not in exclude_columns]
        
        self.log_step("Feature Selection", f"Selected {len(self.feature_names)} features for modeling")
        
        # Ensure no infinite or extremely large values
        for col in self.feature_names:
            if col in self.data.columns:
                # Replace infinite values with NaN, then fill with median
                self.data[col] = self.data[col].replace([np.inf, -np.inf], np.nan)
                if self.data[col].isnull().any():
                    median_value = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_value)
        
        return self.data
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\n SPLITTING DATA (test_size: {test_size})")
        print("-" * 40)
        
        # Prepare features and target
        X = self.data[self.feature_names]
        y = self.data['win_binary']
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Log split information
        train_class_dist = y_train.value_counts()
        test_class_dist = y_test.value_counts()
        
        self.log_step("Data Split", f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        self.log_step("Train Distribution", f"Winners: {train_class_dist.get(1, 0)}, Losers: {train_class_dist.get(0, 0)}")
        self.log_step("Test Distribution", f"Winners: {test_class_dist.get(1, 0)}, Losers: {test_class_dist.get(0, 0)}")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, scale_method='standard', outlier_method='iqr'):
        """Run the complete preprocessing pipeline"""
        print("\n RUNNING COMPLETE PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Clean target variable
        self.clean_target_variable()
        
        # Step 2: Standardize categorical values
        self.standardize_categorical_values()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Feature engineering
        self.feature_engineering()
        
        # Step 5: Encode categorical features
        self.encode_categorical_features()
        
        # Step 6: Handle outliers
        self.handle_outliers(method=outlier_method)
        
        # Step 7: Scale features
        self.scale_features(method=scale_method)
        
        # Step 8: Prepare model features
        self.prepare_model_features()
        
        # Step 9: Split data
        X_train, X_test, y_train, y_test = self.split_data()
        
        print(f"\n PREPROCESSING COMPLETED!")
        print(f" Final dataset shape: {self.data.shape}")
        print(f" Features for modeling: {len(self.feature_names)}")
        print(f" Training samples: {len(X_train)}")
        print(f" Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data and preprocessing objects"""
        print("\n SAVING PROCESSED DATA")
        print("-" * 40)
        
        # Save processed datasets
        X_train.to_csv('outputs/data/X_train.csv', index=False)
        X_test.to_csv('outputs/data/X_test.csv', index=False)
        y_train.to_csv('outputs/data/y_train.csv', index=False)
        y_test.to_csv('outputs/data/y_test.csv', index=False)
        
        # Save preprocessing objects
        preprocessing_objects = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputers': self.imputers,
            'feature_names': self.feature_names
        }
        
        with open('outputs/models/preprocessing_objects.pkl', 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        
        # Save processed full dataset
        self.data.to_csv('outputs/data/processed_data.csv', index=False)
        
        self.log_step("Data Saving", "All processed data and objects saved")
        
        return True
    
    def create_preprocessing_visualizations(self):
        """Create visualizations showing preprocessing effects"""
        print("\n Creating preprocessing visualizations...")
        
        # Create output directory
        os.makedirs('outputs/plots', exist_ok=True)
        
        # Before/After comparison for key features
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Preprocessing Effects: Before vs After', fontsize=16, fontweight='bold')
        
        # Feature distributions before/after
        key_features = ['power_level', 'speed', 'battle_iq'] if all(col in self.original_data.columns for col in ['power_level', 'speed', 'battle_iq']) else self.original_data.select_dtypes(include=[np.number]).columns[:3]
        
        for i, feature in enumerate(key_features):
            if feature in self.original_data.columns and feature in self.data.columns:
                # Before preprocessing
                axes[0, i].hist(self.original_data[feature].dropna(), bins=30, alpha=0.7, color='red', label='Before')
                axes[0, i].set_title(f'{feature} - Before Preprocessing')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].legend()
                
                # After preprocessing
                axes[1, i].hist(self.data[feature].dropna(), bins=30, alpha=0.7, color='green', label='After')
                axes[1, i].set_title(f'{feature} - After Preprocessing')
                axes[1, i].set_xlabel(feature)
                axes[1, i].set_ylabel('Frequency')
                axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig('outputs/plots/05_preprocessing_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Missing values before/after
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Before
        missing_before = self.original_data.isnull().sum()
        missing_before = missing_before[missing_before > 0].sort_values(ascending=False)
        if len(missing_before) > 0:
            missing_before.plot(kind='bar', ax=axes[0], color='red', alpha=0.7)
            axes[0].set_title('Missing Values - Before Preprocessing')
            axes[0].set_ylabel('Missing Count')
            axes[0].tick_params(axis='x', rotation=45)
        
        # After
        missing_after = self.data.isnull().sum()
        missing_after = missing_after[missing_after > 0].sort_values(ascending=False)
        if len(missing_after) > 0:
            missing_after.plot(kind='bar', ax=axes[1], color='green', alpha=0.7)
        else:
            axes[1].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=16, fontweight='bold', color='green')
        
        axes[1].set_title('Missing Values - After Preprocessing')
        axes[1].set_ylabel('Missing Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/06_missing_values_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        print("\n Generating preprocessing report...")
        
        # Create reports directory
        os.makedirs('outputs/reports', exist_ok=True)
        
        report_content = f"""
# Data Preprocessing Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report details the comprehensive data preprocessing pipeline applied to the superhero dataset.

## Original Dataset
- **Shape**: {self.original_data.shape[0]} rows x {self.original_data.shape[1]} columns
- **Memory Usage**: {self.original_data.memory_usage().sum() / 1024**2:.2f} MB
- **Missing Values**: {self.original_data.isnull().sum().sum():,}
- **Duplicate Rows**: {self.original_data.duplicated().sum()}

## Final Processed Dataset
- **Shape**: {self.data.shape[0]} rows x {self.data.shape[1]} columns
- **Features for Modeling**: {len(self.feature_names)}
- **Missing Values**: {self.data.isnull().sum().sum()}
- **Data Types**: All numerical (ready for ML)

## Preprocessing Steps Applied

### 1. Target Variable Cleaning
- Removed rows with missing `win_prob` values
- Capped extreme values to [0, 1] range
- Created binary target variable (`win_binary`)

### 2. Categorical Value Standardization
- Unified role spellings (H3ro -> Hero, VIllain -> Villain)
- Removed extra whitespace and standardized case
- Replaced string 'NaN' values with proper NaN

### 3. Missing Value Handling
- **Strategy**: Median imputation for numerical, mode for categorical
- **High Missing Columns**: Dropped columns with >50% missing values
- **Result**: Zero missing values in final dataset

### 4. Feature Engineering
New features created:
"""
        
        # Add engineered features to report
        engineered_features = [col for col in self.feature_names if any(keyword in col.lower() for keyword in ['ratio', 'efficiency', 'tactical', 'bmi', 'category'])]
        
        for feature in engineered_features:
            report_content += f"- **{feature}**: Combined feature for enhanced predictive power\n"
        
        report_content += f"""

### 5. Categorical Encoding
- **Method**: Label Encoding
- **Columns Encoded**: {len(self.label_encoders)} categorical features
- **Result**: All features converted to numerical format

### 6. Outlier Handling
- **Method**: IQR-based capping (threshold: 1.5)
- **Strategy**: Cap extreme values rather than remove samples
- **Result**: Outliers bounded within reasonable ranges

### 7. Feature Scaling
- **Method**: Standard Scaling (mean=0, std=1)
- **Applied To**: All numerical features except targets
- **Result**: Features normalized for ML algorithms

## Preprocessing Steps Log
"""
        
        # Add preprocessing log
        for i, log_entry in enumerate(self.preprocessing_log, 1):
            report_content += f"{i}. **{log_entry['step']}** ({log_entry['timestamp']}): {log_entry['details']}\n"
        
        report_content += f"""

## Final Feature Set
Total features for modeling: {len(self.feature_names)}

### Feature Categories:
"""
        
        # Categorize features
        basic_features = [f for f in self.feature_names if not any(keyword in f.lower() for keyword in ['ratio', 'efficiency', 'tactical', 'bmi', 'category', 'encoded'])]
        encoded_features = [f for f in self.feature_names if 'encoded' in f.lower()]
        engineered_features = [f for f in self.feature_names if any(keyword in f.lower() for keyword in ['ratio', 'efficiency', 'tactical', 'bmi', 'category'])]
        
        report_content += f"""
- **Basic Numerical Features** ({len(basic_features)}): {', '.join(basic_features[:10])}{'...' if len(basic_features) > 10 else ''}
- **Encoded Categorical Features** ({len(encoded_features)}): {', '.join(encoded_features)}
- **Engineered Features** ({len(engineered_features)}): {', '.join(engineered_features)}

## Quality Assurance
- [x] No missing values remaining
- [x] No infinite values present
- [x] All features are numerical
- [x] Target variable properly prepared
- [x] Class balance maintained in train/test split

## Files Generated
- `X_train.csv`, `X_test.csv`: Training and testing features
- `y_train.csv`, `y_test.csv`: Training and testing targets
- `processed_data.csv`: Complete processed dataset
- `preprocessing_objects.pkl`: Fitted preprocessors for future use
- `05_preprocessing_effects.png`: Before/after visualizations
- `06_missing_values_comparison.png`: Missing value comparisons

## Recommendations for Modeling
1. **Algorithm Selection**: All data is numerical and scaled - suitable for any ML algorithm
2. **Feature Selection**: Consider feature importance analysis to identify top predictors
3. **Class Imbalance**: Monitor if class weights needed during training
4. **Cross-Validation**: Use stratified CV to maintain class distribution

---
*Report generated by Data Preprocessing Module*
"""
        
        # Save report
        with open('outputs/reports/Preprocessing_Report.md', 'w') as f:
            f.write(report_content)
        
        # Save preprocessing summary as JSON for other modules
        preprocessing_summary = {
            'original_shape': [int(self.original_data.shape[0]), int(self.original_data.shape[1])],
            'final_shape': [int(self.data.shape[0]), int(self.data.shape[1])],
            'feature_count': int(len(self.feature_names)),
            'feature_names': list(self.feature_names),
            'preprocessing_steps': int(len(self.preprocessing_log)),
            'missing_values_handled': int(self.original_data.isnull().sum().sum()),
            'engineered_features': int(len(engineered_features)),
            'encoded_features': int(len(encoded_features))
        }
        
        import json
        with open('outputs/reports/preprocessing_summary.json', 'w') as f:
            json.dump(preprocessing_summary, f, indent=2)
        
        print(" Preprocessing report saved to outputs/reports/")

if __name__ == "__main__":
    # Test the preprocessing module
    print(" Testing DataPreprocessor module...")
    
    # Load sample data
    try:
        data = pd.read_csv('data.csv')
        
        # Create output directories
        os.makedirs('outputs/data', exist_ok=True)
        os.makedirs('outputs/models', exist_ok=True)
        os.makedirs('outputs/plots', exist_ok=True)
        os.makedirs('outputs/reports', exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(data)
        
        # Run preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
        
        # Save processed data
        preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
        
        # Create visualizations
        preprocessor.create_preprocessing_visualizations()
        
        # Generate report
        preprocessor.generate_preprocessing_report()
        
        print("reprocessing module test completed!")
        print(f" Final dataset: {X_train.shape[0] + X_test.shape[0]} samples, {X_train.shape[1]} features")
        
    except FileNotFoundError:
        print(" data.csv not found. Please ensure the data file exists.")
    except Exception as e:
        print(f" Error: {str(e)}")