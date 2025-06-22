"""
Project Utilities Module
Handles final report generation, presentation creation, and project utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class ProjectUtils:
    """
    Utility class for project-wide operations and final deliverable generation
    """
    
    def __init__(self):
        self.project_results = {}
        self.presentation_slides = []
        
    def generate_presentation_slides(self, results):
        """Generate presentation-ready slide content"""
        print("\n📊 GENERATING PRESENTATION SLIDES")
        print("=" * 50)
        
        eda = results.get('eda')
        modeler = results.get('modeler')
        task2 = results.get('task2')
        task3 = results.get('task3')
        
        # Create presentation directory
        os.makedirs('outputs/presentation', exist_ok=True)
        
        # Slide 1: EDA Overview
        slide1_content = self._create_eda_slide(eda)
        
        # Slide 2: EDA Insights
        slide2_content = self._create_eda_insights_slide(eda)
        
        # Slide 3: Methodology
        slide3_content = self._create_methodology_slide(modeler)
        
        # Slide 4: Model Evaluation
        slide4_content = self._create_evaluation_slide(modeler)
        
        # Slide 5: Tasks & Discussion
        slide5_content = self._create_discussion_slide(task2, task3, modeler)
        
        # Combine all slides
        presentation_content = f"""# Superhero Winning Probability Prediction
## Group XX Presentation
### {datetime.now().strftime('%B %d, %Y')}

---

{slide1_content}

---

{slide2_content}

---

{slide3_content}

---

{slide4_content}

---

{slide5_content}

---

## Thank You!
### Questions & Discussion

**Contact Information:**
- Project Repository: [\[GitHub Link\]](https://github.com/SaimAkramGill/kddmproject2025.git)
- Email: [muhammad.akram@edu.uni-graz.at]

**Key Takeaways:**
1. Ranking and intelligence are more predictive than raw power
2. {modeler.best_model_name if modeler else 'Random Forest'} achieved best performance
3. Perfect villains win through balanced excellence, not single strengths
4. Data quality and feature engineering are crucial for model success
"""
        
        # Save presentation
        with open('outputs/presentation/Group_125_Presentation.md', 'w', encoding='utf-8') as f:
            f.write(presentation_content)
        
        # Create presentation plots summary
        self._create_presentation_plots(results)
        
        print(" Presentation slides generated")
        return presentation_content
    
    def _create_presentation_plots(self, results):
        """Create summary plots for presentation"""
        print(" Creating presentation summary plots...")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle('Superhero Winning Probability Prediction - Project Summary', 
                    fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Dataset overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        if results.get('eda'):
            eda = results['eda']
            if hasattr(eda, 'data') and 'win_binary' in eda.data.columns:
                win_counts = eda.data['win_binary'].value_counts()
                ax1.pie(win_counts.values, labels=['Losers', 'Winners'], autopct='%1.1f%%', 
                       colors=['lightcoral', 'lightgreen'])
                ax1.set_title('Dataset Class Distribution')
            else:
                ax1.text(0.5, 0.5, 'Dataset overview\nnot available', ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Model performance (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if results.get('modeler') and hasattr(results['modeler'], 'results'):
            modeler = results['modeler']
            models = list(modeler.results.keys())
            aucs = [modeler.results[m]['test_auc'] for m in models]
            
            bars = ax2.bar(range(len(models)), aucs, color='steelblue', alpha=0.7)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.set_ylabel('Test AUC')
            ax2.set_title('Model Performance Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Highlight best model
            if hasattr(modeler, 'best_model_name'):
                best_idx = models.index(modeler.best_model_name)
                bars[best_idx].set_color('gold')
        else:
            ax2.text(0.5, 0.5, 'Model performance\nnot available', ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Feature importance (middle left)
        ax3 = fig.add_subplot(gs[1:3, :2])
        if results.get('modeler') and hasattr(results['modeler'], 'feature_importance') and results['modeler'].feature_importance is not None:
            top_features = results['modeler'].feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'], color='orange', alpha=0.7)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Most Important Features')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Task 2 predictions (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if results.get('task2') and hasattr(results['task2'], 'character_predictions') and results['task2'].character_predictions is not None:
            task2 = results['task2']
            char_names = task2.character_predictions['name']
            probs = task2.character_predictions['predicted_win_probability']
            colors = ['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                     for role in task2.character_predictions['role']]
            
            bars = ax4.bar(range(len(char_names)), probs, color=colors, alpha=0.7)
            ax4.set_xticks(range(len(char_names)))
            ax4.set_xticklabels(char_names, rotation=45, ha='right')
            ax4.set_ylabel('Win Probability')
            ax4.set_title('Task 2: Character Predictions')
            ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Task 2 predictions\nnot available', ha='center', va='center', transform=ax4.transAxes)
        
        # Plot 5: Task 3 analysis (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:])
        if results.get('task3') and hasattr(results['task3'], 'statistical_analysis_results') and results['task3'].statistical_analysis_results:
            task3 = results['task3']
            if 'extreme_features' in task3.statistical_analysis_results:
                extreme_features = task3.statistical_analysis_results['extreme_features'][:5]
                
                if extreme_features:
                    features = [f['feature'] for f in extreme_features]
                    percentiles = [f['percentile'] for f in extreme_features]
                    
                    bars = ax5.bar(range(len(features)), percentiles, color='purple', alpha=0.7)
                    ax5.set_xticks(range(len(features)))
                    ax5.set_xticklabels(features, rotation=45, ha='right')
                    ax5.set_ylabel('Percentile')
                    ax5.set_title(f'Task 3: {task3.perfect_villain["name"] if task3.perfect_villain else "Perfect Villain"} Top Features')
                    ax5.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90th percentile')
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
                else:
                    ax5.text(0.5, 0.5, 'No extreme features\nfound', ha='center', va='center', transform=ax5.transAxes)
            else:
                ax5.text(0.5, 0.5, 'Task 3 analysis\nincomplete', ha='center', va='center', transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, 'Task 3 analysis\nnot available', ha='center', va='center', transform=ax5.transAxes)
        
        # Summary statistics (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = ""
        if results.get('modeler'):
            summary_text += f"Best Model: {results['modeler'].best_model_name if hasattr(results['modeler'], 'best_model_name') else 'N/A'} "
            summary_text += f"(AUC: {results['modeler'].best_auc:.3f})\n" if hasattr(results['modeler'], 'best_auc') else "\n"
        
        if results.get('eda') and hasattr(results['eda'], 'data'):
            summary_text += f"Dataset: {len(results['eda'].data):,} characters, {len(results['eda'].numerical_features)} numerical features\n"
        
        if results.get('task2') and hasattr(results['task2'], 'character_predictions') and results['task2'].character_predictions is not None:
            summary_text += f"Task 2: Predicted {len(results['task2'].character_predictions)} characters\n"
        
        if results.get('task3') and hasattr(results['task3'], 'perfect_villain') and results['task3'].perfect_villain is not None:
            summary_text += f"Task 3: {results['task3'].perfect_villain['name']} wins through balanced excellence\n"
        
        summary_text += f"Key Insight: Intelligence and ranking matter more than raw power for winning"
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.savefig('outputs/presentation/Project_Summary_Dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Presentation plots created")
    
    def create_presentation_plots(self, results):
        """Public method to create presentation plots (for backward compatibility)"""
        return self._create_presentation_plots(results)
    
    def _create_eda_slide(self, eda):
        """Create EDA overview slide content"""
        if not eda:
            return "## Slide 1: Exploratory Data Analysis - Overview\n\n*EDA results not available*"
        
        slide_content = f"""## Slide 1: Exploratory Data Analysis - Overview

### Dataset Characteristics
- **Dataset Size**: {eda.data.shape[0]:,} characters, {eda.data.shape[1]} features
- **Target Variable**: Winning probability (0.22 to 15.20) → Binary classification (win/lose)
- **Feature Types**: {len(eda.numerical_features)} numerical, {len(eda.categorical_features)} categorical features
- **Class Balance**: {eda.data['win_binary'].mean()*100:.1f}% winners, {(1-eda.data['win_binary'].mean())*100:.1f}% losers (imbalanced)

### Key Findings from EDA
- **Missing Data**: High in eye_color (32.4%), secret_code (37.3%), speed (27.2%)
- **Data Quality Issues**: Role spellings inconsistent (Hero vs H3ro vs HerO)
- **Outliers**: Some extreme win_prob values (max: 15.20 vs typical 0.2-1.0)

### Target Distribution
- Winning probability ranges from 0.22 to 15.20 (capped at 1.0 for modeling)
- {eda.data['win_binary'].mean()*100:.1f}% of characters have winning probability > 0.5
- Clear class imbalance favoring winners"""
        
        return slide_content
    
    def _create_eda_insights_slide(self, eda):
        """Create EDA insights slide content"""
        if not eda or not hasattr(eda, 'correlation_matrix'):
            return "## Slide 2: Exploratory Data Analysis - Feature Insights\n\n*Correlation analysis not available*"
        
        # Get top correlations
        if 'win_prob' in eda.correlation_matrix.columns:
            target_corr = eda.correlation_matrix['win_prob'].abs().sort_values(ascending=False)
            top_features = target_corr.head(6)[1:]  # Exclude target itself
        else:
            top_features = pd.Series()
        
        slide_content = f"""## Slide 2: Exploratory Data Analysis - Feature Insights

### Most Important Features (Correlation with Win Probability)
"""
        
        if len(top_features) > 0:
            for i, (feature, corr) in enumerate(top_features.items(), 1):
                direction = "positive" if eda.correlation_matrix.loc[feature, 'win_prob'] > 0 else "negative"
                interpretation = {
                    'ranking': 'lower ranking = higher win probability',
                    'battle_iq': 'higher battle IQ = higher win probability',
                    'intelligence': 'smarter characters tend to win',
                    'speed': 'faster characters have slight advantage',
                    'height': 'shorter characters tend to win slightly more'
                }.get(feature, f'{direction} correlation with winning')
                
                slide_content += f"{i}. **{feature}**: r = {eda.correlation_matrix.loc[feature, 'win_prob']:+.4f} ({interpretation})\n"
        
        slide_content += f"""

### Notable Patterns Discovered
- **Character Roles**: Heroes ({eda.data['role'].value_counts().get('Hero', 0)}) vs Villains ({eda.data['role'].value_counts().get('Villain', 0)}) roughly balanced, but multiple spellings
- **Power vs Success**: Surprisingly weak correlation between raw power and winning
- **Intelligence Matters**: Battle IQ and intelligence are key predictors
- **Ranking is King**: Most important single predictor (popularity/effectiveness measure)"""
        
        return slide_content
    
    def _create_methodology_slide(self, modeler):
        """Create methodology slide content"""
        if not modeler:
            return "## Slide 3: Methodology\n\n*Modeling results not available*"
        
        slide_content = f"""## Slide 3: Methodology

### Data Preprocessing Pipeline
- **Missing Value Imputation**: Median imputation for numerical features, mode for categorical
- **Categorical Encoding**: Label encoding for categorical variables
- **Role Standardization**: Unified Hero/Villain spellings (H3ro → Hero, VIllain → Villain)
- **Target Engineering**: Capped extreme values, binary classification (threshold = 0.5)

### Data Splitting Strategy
- **Training Set**: 80% of data ({len(modeler.X_train):,} samples)
- **Test Set**: 20% of data ({len(modeler.X_test):,} samples)  
- **Stratified Split**: Maintained {modeler.y_train.mean()*100:.1f}% winner / {(1-modeler.y_train.mean())*100:.1f}% loser balance
- **Cross-Validation**: 5-fold stratified CV for model selection

### Models Evaluated
- **Random Forest**: Tree-based ensemble with class balancing
- **Gradient Boosting**: Sequential boosting for complex patterns  
- **Logistic Regression**: Linear baseline with feature scaling
- **SVM**: Support Vector Machine with RBF kernel
- **AdaBoost**: Adaptive boosting ensemble
- **K-Nearest Neighbors**: Distance-based classification"""
        
        return slide_content
    
    def _create_evaluation_slide(self, modeler):
        """Create model evaluation slide content"""
        if not modeler or not modeler.results:
            return "## Slide 4: Model Evaluation\n\n*Model results not available*"
        
        # Get results sorted by performance
        sorted_results = sorted(modeler.results.items(), key=lambda x: x[1]['test_auc'], reverse=True)
        
        slide_content = f"""## Slide 4: Model Evaluation

### Model Performance Comparison
| Model | Cross-Val AUC | Test AUC | Best Model |
|-------|---------------|----------|------------|
"""
        
        for name, result in sorted_results:
            is_best = "✓" if name == modeler.best_model_name else "✗"
            slide_content += f"| {name} | {result['cv_auc_mean']:.3f} | {result['test_auc']:.3f} | {is_best} |\n"
        
        slide_content += f"""

### Best Model: {modeler.best_model_name}
- **Final Test AUC**: {modeler.best_auc:.3f} - {'Excellent' if modeler.best_auc > 0.8 else 'Good' if modeler.best_auc > 0.7 else 'Fair'} performance given weak feature correlations
- **Precision**: {modeler.results[modeler.best_model_name]['test_precision']:.3f} - {modeler.results[modeler.best_model_name]['test_precision']*100:.0f}% of predicted winners actually win  
- **Recall**: {modeler.results[modeler.best_model_name]['test_recall']:.3f} - Captures {modeler.results[modeler.best_model_name]['test_recall']*100:.0f}% of actual winners
- **F1-Score**: {modeler.results[modeler.best_model_name]['test_f1']:.3f} - Balanced performance metric

### Feature Importance (Top 5)"""
        
        if hasattr(modeler, 'feature_importance') and modeler.feature_importance is not None:
            for i, (_, row) in enumerate(modeler.feature_importance.head(5).iterrows(), 1):
                interpretation = {
                    'ranking': 'Most predictive single feature',
                    'battle_iq': 'Combat intelligence crucial', 
                    'intelligence': 'General intelligence matters',
                    'power_level': 'Raw power still important',
                    'speed': 'Agility provides advantage'
                }.get(row['feature'], 'Important predictor')
                
                slide_content += f"\n{i}. **{row['feature']}**: {row['importance']:.3f} - {interpretation}"
        
        return slide_content
    
    def _create_discussion_slide(self, task2, task3, modeler):
        """Create discussion slide with task results"""
        slide_content = f"""## Slide 5: Discussion & Task Results

### Task 2: Unseen Data Predictions
"""
        
        if task2 and task2.character_predictions is not None:
            # Separate heroes and villains
            heroes = task2.character_predictions[task2.character_predictions['role'].isin(['Hero', 'H3ro', 'HerO'])]
            villains = task2.character_predictions[task2.character_predictions['role'].isin(['Villain', 'VIllain', 'VillaIn'])]
            
            slide_content += "**Three Superheroes:**\n"
            for _, hero in heroes.iterrows():
                slide_content += f"- 🦸 {hero['name']}: {hero['predicted_win_probability']:.3f} ({hero['predicted_win_probability']*100:.1f}%)\n"
            
            slide_content += "\n**Three Villains:**\n"
            for _, villain in villains.iterrows():
                slide_content += f"- 🦹 {villain['name']}: {villain['predicted_win_probability']:.3f} ({villain['predicted_win_probability']*100:.1f}%)\n"
            
            if task2.fight_predictions is not None:
                slide_content += "\n**Fight Predictions:**\n"
                for _, fight in task2.fight_predictions.iterrows():
                    slide_content += f"- Fight {fight['fight_number']}: {fight['fighter1']} vs {fight['fighter2']} → **{fight['winner']} WINS** ({fight['winner_prob']:.3f} vs {fight['loser_prob']:.3f})\n"
        else:
            slide_content += "*Task 2 predictions not available*\n"
        
        slide_content += f"""
### Task 3: Perfect Villain Analysis (100% Win Rate)
"""
        
        if task3 and task3.perfect_villain is not None:
            slide_content += f"""**Why {task3.perfect_villain['name']} Always Wins:**
- **Power Level**: {task3.perfect_villain.get('power_level', 'N/A'):,} (91st percentile) - Near maximum power
- **Excellent Ranking**: {task3.perfect_villain.get('ranking', 'N/A'):,} (23rd percentile) - Top-tier popularity/effectiveness  
- **Extensive Training**: {task3.perfect_villain.get('training_time', 'N/A'):,} hours (92nd percentile) - Exceptional preparation
- **Balanced Stats**: Above average in speed, intelligence, and battle IQ

**Key Insight**: {task3.perfect_villain['name']} doesn't dominate any single category but achieves an optimal combination of high power, excellent ranking, and extensive training - creating an unbeatable profile.
"""
        else:
            slide_content += "*Task 3 analysis not available*\n"
        
        slide_content += f"""
### Special Observations
- **Surprising Finding**: Raw power alone doesn't guarantee victory - ranking and intelligence matter more
- **Model Limitations**: Weak correlations suggest other unmeasured factors influence winning
- **Data Quality**: Inconsistent spellings and missing values indicate synthetic data artifacts
- **Real-World Applicability**: Framework could apply to sports analytics, gaming balance, or competitive rankings"""
        
        return slide_content
    
    def _create_presentation_plots(self, results):
        """Create summary plots for presentation"""
        print("🎨 Creating presentation summary plots...")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Main title
        fig.suptitle('Superhero Winning Probability Prediction - Project Summary', 
                    fontsize=20, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Dataset overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        if results.get('eda'):
            eda = results['eda']
            win_counts = eda.data['win_binary'].value_counts()
            ax1.pie(win_counts.values, labels=['Losers', 'Winners'], autopct='%1.1f%%', 
                   colors=['lightcoral', 'lightgreen'])
            ax1.set_title('Dataset Class Distribution')
        
        # Plot 2: Model performance (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if results.get('modeler'):
            modeler = results['modeler']
            models = list(modeler.results.keys())
            aucs = [modeler.results[m]['test_auc'] for m in models]
            
            bars = ax2.bar(range(len(models)), aucs, color='steelblue', alpha=0.7)
            ax2.set_xticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.set_ylabel('Test AUC')
            ax2.set_title('Model Performance Comparison')
            ax2.grid(True, alpha=0.3)
            
            # Highlight best model
            best_idx = models.index(modeler.best_model_name)
            bars[best_idx].set_color('gold')
        
        # Plot 3: Feature importance (middle left)
        ax3 = fig.add_subplot(gs[1:3, :2])
        if results.get('modeler') and hasattr(results['modeler'], 'feature_importance'):
            top_features = results['modeler'].feature_importance.head(10)
            ax3.barh(range(len(top_features)), top_features['importance'], color='orange', alpha=0.7)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['feature'])
            ax3.set_xlabel('Feature Importance')
            ax3.set_title('Top 10 Most Important Features')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Task 2 predictions (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if results.get('task2') and results['task2'].character_predictions is not None:
            task2 = results['task2']
            char_names = task2.character_predictions['name']
            probs = task2.character_predictions['predicted_win_probability']
            colors = ['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                     for role in task2.character_predictions['role']]
            
            bars = ax4.bar(range(len(char_names)), probs, color=colors, alpha=0.7)
            ax4.set_xticks(range(len(char_names)))
            ax4.set_xticklabels(char_names, rotation=45, ha='right')
            ax4.set_ylabel('Win Probability')
            ax4.set_title('Task 2: Character Predictions')
            ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Task 3 analysis (bottom right)
        ax5 = fig.add_subplot(gs[2, 2:])
        if results.get('task3') and results['task3'].statistical_analysis_results:
            task3 = results['task3']
            extreme_features = task3.statistical_analysis_results['extreme_features'][:5]
            
            if extreme_features:
                features = [f['feature'] for f in extreme_features]
                percentiles = [f['percentile'] for f in extreme_features]
                
                bars = ax5.bar(range(len(features)), percentiles, color='purple', alpha=0.7)
                ax5.set_xticks(range(len(features)))
                ax5.set_xticklabels(features, rotation=45, ha='right')
                ax5.set_ylabel('Percentile')
                ax5.set_title(f'Task 3: {task3.perfect_villain["name"]} Top Features')
                ax5.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90th percentile')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # Summary statistics (bottom)
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = ""
        if results.get('modeler'):
            summary_text += f"Best Model: {results['modeler'].best_model_name} (AUC: {results['modeler'].best_auc:.3f})\n"
        
        if results.get('eda'):
            summary_text += f"Dataset: {len(results['eda'].data):,} characters, {len(results['eda'].numerical_features)} numerical features\n"
        
        if results.get('task2'):
            summary_text += f"Task 2: Predicted {len(results['task2'].character_predictions) if results['task2'].character_predictions is not None else 0} characters\n"
        
        if results.get('task3'):
            summary_text += f"Task 3: {results['task3'].perfect_villain['name'] if results['task3'].perfect_villain is not None else 'N/A'} wins through balanced excellence\n"
        
        summary_text += f"Key Insight: Intelligence and ranking matter more than raw power for winning"
        
        ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.savefig('outputs/presentation/Project_Summary_Dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Presentation plots created")
    
    def generate_executive_summary(self, results):
        """Generate executive summary document"""
        print("\n📋 Generating executive summary...")
        
        summary_content = f"""
# Superhero Winning Probability Prediction - Executive Summary
**Project Completion Date**: {datetime.now().strftime('%B %d, %Y')}

## Project Overview
This machine learning project analyzed {results['eda'].data.shape[0] if results.get('eda') else 'N/A'} superhero and villain characters to predict winning probabilities and understand the factors that determine success in combat scenarios.

## Key Findings

### 1. Model Performance
- **Best Algorithm**: {results['modeler'].best_model_name if results.get('modeler') else 'N/A'}
- **Test Accuracy**: {results['modeler'].best_auc:.3f} AUC score
- **Key Insight**: Ensemble methods outperformed linear models

### 2. Most Important Success Factors
"""
        
        if results.get('modeler') and hasattr(results['modeler'], 'feature_importance'):
            for i, (_, row) in enumerate(results['modeler'].feature_importance.head(5).iterrows(), 1):
                summary_content += f"{i}. **{row['feature']}** (importance: {row['importance']:.3f})\n"
        
        summary_content += f"""

### 3. Character Analysis Results
- **Task 2**: Successfully predicted winning probabilities for 6 new characters
- **Task 3**: Identified why {results['task3'].perfect_villain['name'] if results.get('task3') and results['task3'].perfect_villain is not None else 'the perfect villain'} achieves 100% win rate

### 4. Surprising Discoveries
- Raw power is less predictive than intelligence and strategic ranking
- Character popularity (ranking) is the strongest single predictor
- Perfect characters win through balanced excellence, not single strengths

## Business Implications
1. **Character Development**: Focus on intelligence and strategic positioning over raw power
2. **Competitive Analysis**: Use ranking and training metrics for performance prediction  
3. **Data Quality**: Address inconsistent categorical values for better model performance

## Technical Achievements
- Comprehensive data preprocessing pipeline handling missing values and outliers
- Systematic model comparison across 6 different algorithms
- Statistical analysis revealing key performance drivers
- Automated visualization and reporting system

## Recommendations
1. **Immediate**: Deploy {results['modeler'].best_model_name if results.get('modeler') else 'the best model'} for character evaluation
2. **Short-term**: Collect additional training and combat strategy data
3. **Long-term**: Develop ensemble approaches combining multiple character aspects

---
*Generated by Superhero Prediction Analysis System*
"""
        
        # Save executive summary
        with open('outputs/presentation/Executive_Summary.md', 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(" Executive summary generated")
        return summary_content
    
    def generate_final_report(self, results):
        """Generate comprehensive final project report"""
        print("\n Generating final project report...")
        
        report_content = f"""
# Superhero Winning Probability Prediction - Final Project Report
**Course**: Machine Learning / Data Science
**Team**: Group 125
**Date**: {datetime.now().strftime('%B %d, %Y')}

## Table of Contents
1. [Introduction](#introduction)
2. [Data Analysis](#data-analysis)  
3. [Methodology](#methodology)
4. [Results](#results)
5. [Task Solutions](#task-solutions)
6. [Discussion](#discussion)
7. [Conclusions](#conclusions)
8. [Appendices](#appendices)

## 1. Introduction

### Project Objective
The goal of this project was to develop machine learning models to predict the winning probabilities of superhero and villain characters based on their attributes, and to analyze the factors that contribute to combat success.

### Dataset Description
- **Size**: {results['eda'].data.shape[0] if results.get('eda') else 'N/A'} characters with {results['eda'].data.shape[1] if results.get('eda') else 'N/A'} features
- **Target Variable**: Winning probability (continuous) converted to binary classification
- **Features**: Mix of numerical (power, speed, intelligence) and categorical (role, universe) attributes

### Research Questions
1. What character attributes most strongly predict winning probability?
2. Can we accurately predict outcomes for unseen character matchups?
3. What makes a character achieve perfect (100%) win rate?

## 2. Data Analysis

### 2.1 Exploratory Data Analysis
- **Class Distribution**: {results['eda'].data['win_binary'].mean()*100:.1f}% winners vs {(1-results['eda'].data['win_binary'].mean())*100:.1f}% losers
- **Data Quality Issues**: Missing values in eye_color (32.4%), inconsistent role spellings
- **Feature Correlations**: Weak to moderate correlations with target variable

### 2.2 Key Patterns Discovered
"""
        
        if results.get('eda') and hasattr(results['eda'], 'correlation_matrix'):
            if 'win_prob' in results['eda'].correlation_matrix.columns:
                target_corr = results['eda'].correlation_matrix['win_prob'].abs().sort_values(ascending=False)
                top_3 = target_corr.head(4)[1:]  # Exclude target itself, get top 3
                for feature, corr in top_3.items():
                    direction = "positively" if results['eda'].correlation_matrix.loc[feature, 'win_prob'] > 0 else "negatively"
                    report_content += f"- **{feature}** correlates {direction} with winning (r = {results['eda'].correlation_matrix.loc[feature, 'win_prob']:+.3f})\n"
        
        report_content += f"""

## 3. Methodology

### 3.1 Data Preprocessing
1. **Missing Value Treatment**: Median imputation for numerical, mode for categorical
2. **Categorical Encoding**: Label encoding with handling for unseen categories
3. **Feature Engineering**: Created composite features (ratios, efficiency scores)
4. **Outlier Handling**: IQR-based capping to preserve data points
5. **Feature Scaling**: StandardScaler normalization

### 3.2 Model Development
**Algorithms Evaluated**:
"""
        
        if results.get('modeler'):
            for name, result in results['modeler'].results.items():
                report_content += f"- {name}: {result['test_auc']:.3f} AUC\n"
        
        report_content += f"""

**Model Selection Criteria**:
- Primary: ROC-AUC score (handles class imbalance)
- Secondary: Cross-validation stability
- Tertiary: Interpretability and feature importance

### 3.3 Evaluation Framework
- **Cross-Validation**: 5-fold stratified CV
- **Metrics**: AUC, Precision, Recall, F1-Score
- **Validation**: Hold-out test set (20% of data)

## 4. Results

### 4.1 Model Performance
**Best Model**: {results['modeler'].best_model_name if results.get('modeler') else 'N/A'}
- **Test AUC**: {results['modeler'].best_auc:.3f}
- **Cross-Val AUC**: {results['modeler'].results[results['modeler'].best_model_name]['cv_auc_mean']:.3f} (±{results['modeler'].results[results['modeler'].best_model_name]['cv_auc_std']:.3f})
- **Test Accuracy**: {results['modeler'].results[results['modeler'].best_model_name]['test_accuracy']:.3f}

### 4.2 Feature Importance Analysis
"""
        
        if results.get('modeler') and hasattr(results['modeler'], 'feature_importance'):
            report_content += "**Top 5 Most Important Features**:\n"
            for i, (_, row) in enumerate(results['modeler'].feature_importance.head(5).iterrows(), 1):
                report_content += f"{i}. {row['feature']}: {row['importance']:.4f}\n"
        
        report_content += f"""

## 5. Task Solutions

### 5.1 Task 2: Character Predictions
**Objective**: Predict winning probabilities for 6 unseen characters

**Results**:
"""
        
        if results.get('task2') and results['task2'].character_predictions is not None:
            for _, char in results['task2'].character_predictions.iterrows():
                char_type = "Hero" if char['role'] in ['Hero', 'H3ro', 'HerO'] else "Villain"
                report_content += f"- {char_type} {char['name']}: {char['predicted_win_probability']:.3f} win probability\n"
            
            if results['task2'].fight_predictions is not None:
                report_content += f"\n**Fight Outcome Predictions**:\n"
                for _, fight in results['task2'].fight_predictions.iterrows():
                    report_content += f"- {fight['fighter1']} vs {fight['fighter2']}: **{fight['winner']} wins** (margin: {fight['margin']:.3f})\n"
        
        report_content += f"""

### 5.2 Task 3: Perfect Villain Analysis  
**Objective**: Explain why a specific villain has 100% win rate

**Subject**: {results['task3'].perfect_villain['name'] if results.get('task3') and results['task3'].perfect_villain is not None else 'Perfect Villain'}

**Key Findings**:
"""
        
        if results.get('task3') and results['task3'].statistical_analysis_results:
            extreme_features = results['task3'].statistical_analysis_results['extreme_features'][:3]
            for feature_data in extreme_features:
                report_content += f"- **{feature_data['feature']}**: {feature_data['percentile']:.1f}th percentile (Z-score: {feature_data['z_score']:+.2f})\n"
        
        report_content += f"""

**Explanation**: The perfect villain achieves 100% win rate through optimal combination of high power, excellent strategic ranking, and extensive training rather than dominating any single attribute.

## 6. Discussion

### 6.1 Key Insights
1. **Intelligence Over Power**: Mental attributes (intelligence, battle_iq) are more predictive than physical attributes
2. **Strategic Importance**: Character ranking (popularity/reputation) is the strongest single predictor
3. **Balanced Excellence**: Perfect characters excel across multiple dimensions rather than specializing

### 6.2 Model Limitations
- Weak feature correlations suggest unmeasured factors influence winning
- Synthetic data may not capture real combat dynamics  
- Class imbalance requires careful evaluation interpretation

### 6.3 Real-World Applications
- **Sports Analytics**: Player performance prediction
- **Game Design**: Character balancing in video games
- **Competitive Analysis**: Strategic planning in competitive scenarios

## 7. Conclusions

### 7.1 Research Question Answers
1. **Most Predictive Attributes**: Ranking, battle_iq, intelligence, speed (in order of importance)
2. **Prediction Accuracy**: Achieved {results['modeler'].best_auc:.3f} AUC, indicating good discriminative ability
3. **Perfect Win Rate Formula**: Balanced high performance across power, strategy, and preparation

### 7.2 Project Success Metrics
 Successfully built predictive model with good performance
 Identified key success factors through feature importance analysis  
 Completed both prediction tasks with actionable insights
 Delivered comprehensive analysis with visualizations and reports

### 7.3 Future Recommendations
1. **Data Collection**: Gather combat scenario and strategy data
2. **Feature Engineering**: Develop interaction terms between key features
3. **Model Enhancement**: Experiment with deep learning approaches
4. **Validation**: Test on real-world competitive data

## 8. Appendices

### Appendix A: Technical Implementation
- **Programming Language**: Python 3.x
- **Key Libraries**: scikit-learn, pandas, matplotlib, plotly
- **Model Architecture**: Ensemble methods with cross-validation
- **Code Repository**: [Available upon request]

### Appendix B: File Deliverables
- `main.py`: Complete analysis pipeline
- `EDA.py`: Exploratory data analysis module  
- `modeling.py`: Machine learning implementation
- `task2.py`: Character prediction module
- `task3.py`: Perfect villain analysis module
- `outputs/`: All results, plots, and reports

### Appendix C: Detailed Results
Comprehensive results including confusion matrices, ROC curves, and statistical analyses are available in the outputs directory.

---

"""
        
        # Save final report
        with open('outputs/presentation/Final_Project_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("Final project report generated")
        return report_content
    
    def create_project_archive(self):
        """Create a zip archive with all project deliverables"""
        print("\n Creating project archive...")
        
        import zipfile
        from pathlib import Path
        
        # Create archive
        archive_name = f"Group_XX_Superhero_Prediction_{datetime.now().strftime('%Y%m%d')}.zip"
        archive_path = f"outputs/{archive_name}"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all Python files
            python_files = ['main.py', 'EDA.py', 'preprocessing.py', 'modeling.py', 'task2.py', 'task3.py', 'utils.py']
            for file in python_files:
                if os.path.exists(file):
                    zipf.write(file, f"source_code/{file}")
            
            # Add outputs directory
            outputs_dir = Path('outputs')
            if outputs_dir.exists():
                for file_path in outputs_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = str(file_path).replace('outputs/', '')
                        zipf.write(file_path, f"outputs/{arcname}")
            
            # Add data files
            data_files = ['data.csv', 'Task2_superheroes_villains.csv', 'Task2_matches.csv', 'Task3_villain.csv']
            for file in data_files:
                if os.path.exists(file):
                    zipf.write(file, f"data/{file}")
            
            # Add README
            readme_content = f"""# Superhero Winning Probability Prediction
Group XX - Final Project Submission

## Project Structure
- `source_code/`: All Python modules
- `data/`: Original datasets
- `outputs/`: All generated results
  - `plots/`: Visualizations and charts
  - `reports/`: Analysis reports and summaries
  - `models/`: Trained models and metadata
  - `predictions/`: Task 2 and Task 3 results
  - `presentation/`: Final presentation materials

## Quick Start
1. Run `python main.py` for complete analysis
2. Check `outputs/presentation/` for final deliverables
3. View `Final_Project_Report.md` for comprehensive results

## Key Results
- Best Model: {self.project_results.get('best_model', 'Random Forest')}
- Test AUC: {self.project_results.get('best_auc', 'N/A')}
- Task 2: Character predictions completed
- Task 3: Perfect villain analysis completed

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            zipf.writestr('README.md', readme_content)
        
        print(f" Project archive created: {archive_path}")
        return archive_path
    
    def generate_presentation_summary(self, results):
        """Generate a concise presentation summary"""
        print("\n Generating presentation summary...")
        
        summary = {
            'project_title': 'Superhero Winning Probability Prediction',
            'completion_date': datetime.now().strftime('%Y-%m-%d'),
            'dataset_size': results['eda'].data.shape[0] if results.get('eda') else None,
            'best_model': results['modeler'].best_model_name if results.get('modeler') else None,
            'best_auc': float(results['modeler'].best_auc) if results.get('modeler') else None,
            'top_features': [],
            'task2_completed': results.get('task2') is not None,
            'task3_completed': results.get('task3') is not None,
            'key_insights': [
                "Intelligence and ranking are more predictive than raw power",
                "Perfect characters win through balanced excellence",
                "Data quality significantly impacts model performance",
                "Ensemble methods outperform linear approaches"
            ]
        }
        
        # Add top features
        if results.get('modeler') and hasattr(results['modeler'], 'feature_importance'):
            summary['top_features'] = [
                {
                    'feature': row['feature'],
                    'importance': float(row['importance'])
                }
                for _, row in results['modeler'].feature_importance.head(5).iterrows()
            ]
        
        # Save summary
        with open('outputs/presentation/presentation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        # Store for archive
        self.project_results = summary
        
        print(" Presentation summary generated")
        return summary
    
    def validate_deliverables(self):
        """Validate that all required deliverables are present"""
        print("\nVALIDATING PROJECT DELIVERABLES")
        print("=" * 50)
        
        required_files = {
            'Source Code': [
                'main.py',
                'EDA.py', 
                'preprocessing.py',
                'modeling.py',
                'task2.py',
                'task3.py',
                'utils.py'
            ],
            'Data Files': [
                'data.csv',
                'Task2_superheroes_villains.csv',
                'Task2_matches.csv', 
                'Task3_villain.csv'
            ],
            'Reports': [
                'outputs/reports/EDA_Report.md',
                'outputs/reports/Modeling_Report.md',
                'outputs/reports/Task2_Report.md',
                'outputs/reports/Task3_Report.md'
            ],
            'Visualizations': [
                'outputs/plots/01_dataset_overview.png',
                'outputs/plots/07_model_comparison.png',
                'outputs/plots/14_task2_character_predictions.png',
                'outputs/plots/17_task3_statistical_analysis.png'
            ],
            'Presentation': [
                'outputs/presentation/Group_XX_Presentation.md',
                'outputs/presentation/Executive_Summary.md',
                'outputs/presentation/Final_Project_Report.md'
            ],
            'Models': [
                'outputs/models/model_metadata.json',
                'outputs/models/preprocessing_objects.pkl'
            ]
        }
        
        validation_results = {}
        all_present = True
        
        for category, files in required_files.items():
            category_results = {}
            category_complete = True
            
            for file in files:
                exists = os.path.exists(file)
                category_results[file] = exists
                if not exists:
                    category_complete = False
                    all_present = False
            
            validation_results[category] = {
                'files': category_results,
                'complete': category_complete,
                'completion_rate': sum(category_results.values()) / len(category_results)
            }
            
            # Display results
            status = "" if category_complete else "⚠️"
            completion = validation_results[category]['completion_rate'] * 100
            print(f"{status} {category}: {completion:.0f}% complete")
            
            for file, exists in category_results.items():
                file_status = "" if exists else "❌"
                print(f"   {file_status} {file}")
        
        # Overall status
        print(f"\n{' ALL DELIVERABLES COMPLETE' if all_present else '⚠️ SOME DELIVERABLES MISSING'}")
        
        # Save validation report
        with open('outputs/validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results
    
    def cleanup_outputs(self):
        """Clean up and organize output files"""
        print("\n🧹 Cleaning up output files...")
        
        # Remove any temporary files
        temp_patterns = ['*.tmp', '*.temp', '*~', '.DS_Store']
        
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            for pattern in temp_patterns:
                for temp_file in outputs_dir.rglob(pattern):
                    try:
                        temp_file.unlink()
                        print(f"Removed: {temp_file}")
                    except:
                        pass
        
        # Organize files by type
        # (This could include moving misplaced files to correct directories)
        
        print(" Output cleanup completed")

def create_project_structure():
    """Create the complete project directory structure"""
    directories = [
        'outputs',
        'outputs/plots',
        'outputs/reports', 
        'outputs/models',
        'outputs/predictions',
        'outputs/presentation',
        'outputs/data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(" Project structure created!")

if __name__ == "__main__":
    # Test the utils module
    print(" Testing ProjectUtils module...")
    
    # Create project structure
    create_project_structure()
    
    # Initialize utils
    utils = ProjectUtils()
    
    # Test validation
    validation_results = utils.validate_deliverables()
    
    print(" Utils module test completed!")
    print(" Utils module ready for integration with main pipeline")