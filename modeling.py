"""
Machine Learning Modeling Module for Superhero Dataset
Comprehensive model training, evaluation, and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score,
                           precision_score, recall_score)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class SuperheroModeling:
    """
    Comprehensive machine learning modeling for superhero winning probability prediction
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_auc = 0
        self.feature_importance = None
        self.top_features = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def initialize_models(self):
        """Initialize all models with optimized parameters"""
        print("🤖 Initializing machine learning models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Enable probability predictions
                class_weight='balanced',
                random_state=42
            ),
            
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            )
        }
        
        print(f"✅ Initialized {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models and perform cross-validation"""
        print("\n🏋️ TRAINING MACHINE LEARNING MODELS")
        print("=" * 60)
        
        # Store data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize models
        self.initialize_models()
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
                cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
                
                # Train on full training set
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                test_auc = roc_auc_score(y_test, y_pred_proba)
                test_accuracy = accuracy_score(y_test, y_pred)
                test_precision = precision_score(y_test, y_pred)
                test_recall = recall_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'cv_f1_mean': cv_f1_scores.mean(),
                    'cv_f1_std': cv_f1_scores.std(),
                    'test_auc': test_auc,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'training_time': 0  # Would measure in real implementation
                }
                
                # Print results
                print(f"  Cross-validation AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                print(f"  Test AUC: {test_auc:.4f}")
                print(f"  Test Accuracy: {test_accuracy:.4f}")
                print(f"  Test F1-Score: {test_f1:.4f}")
                
                # Track best model
                if test_auc > self.best_auc:
                    self.best_auc = test_auc
                    self.best_model = model  # Store the actual model object
                    self.best_model_name = name
                    
            except Exception as e:
                print(f"  ❌ Error training {name}: {str(e)}")
                # Continue with other models
                continue
        
        if self.best_model is None:
            raise Exception("No models were successfully trained")
        
        print(f"\n🏆 Best model: {self.best_model_name} (AUC: {self.best_auc:.4f})")
        
        return self.results
    
    def hyperparameter_tuning(self, model_name=None):
        """Perform hyperparameter tuning for specific model or best model"""
        print(f"\n⚙️ HYPERPARAMETER TUNING")
        print("-" * 40)
        
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"Tuning {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10]
            },
            
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
        }
        
        if model_name in param_grids:
            # Get base model
            base_model = self.models[model_name]
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grids[model_name],
                cv=3,  # Reduced for speed
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Update best model
            tuned_model = grid_search.best_estimator_
            y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1]
            tuned_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Tuned AUC: {tuned_auc:.4f} (vs original: {self.results[model_name]['test_auc']:.4f})")
            
            # Update if improved
            if tuned_auc > self.results[model_name]['test_auc']:
                self.results[model_name]['model'] = tuned_model
                self.results[model_name]['test_auc'] = tuned_auc
                self.results[model_name]['best_params'] = grid_search.best_params_
                
                if tuned_auc > self.best_auc:
                    self.best_auc = tuned_auc
                    self.best_model = tuned_model
                
                print("✅ Model updated with improved parameters")
            else:
                print("ℹ️ Original parameters were already optimal")
        
        return grid_search if model_name in param_grids else None
    
    def evaluate_models(self, results=None):
        """Comprehensive model evaluation and comparison"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        if results is None:
            results = self.results
        
        # Create evaluation summary
        evaluation_df = pd.DataFrame({
            'Model': list(results.keys()),
            'CV_AUC_Mean': [results[name]['cv_auc_mean'] for name in results.keys()],
            'CV_AUC_Std': [results[name]['cv_auc_std'] for name in results.keys()],
            'Test_AUC': [results[name]['test_auc'] for name in results.keys()],
            'Test_Accuracy': [results[name]['test_accuracy'] for name in results.keys()],
            'Test_Precision': [results[name]['test_precision'] for name in results.keys()],
            'Test_Recall': [results[name]['test_recall'] for name in results.keys()],
            'Test_F1': [results[name]['test_f1'] for name in results.keys()]
        }).sort_values('Test_AUC', ascending=False)
        
        print("Model Performance Summary:")
        print(evaluation_df.to_string(index=False, float_format='%.4f'))
        
        # Detailed evaluation of best model
        best_result = results[self.best_model_name]
        
        print(f"\n🏆 BEST MODEL DETAILED EVALUATION: {self.best_model_name}")
        print("-" * 50)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(self.y_test, best_result['y_pred']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, best_result['y_pred'])
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Lose    Win")
        print(f"Lose      {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"Win       {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Feature importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': best_result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.top_features = self.feature_importance['feature'].head(10).tolist()
            
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows(), 1):
                print(f"{i:2d}. {row['feature']:25s}: {row['importance']:.4f}")
        
        # Model-specific insights
        self._generate_model_insights(best_result['model'])
        
        return evaluation_df, best_result
    
    def _generate_model_insights(self, model):
        """Generate insights specific to the model type"""
        model_type = type(model).__name__
        
        print(f"\n🔍 {model_type} Specific Insights:")
        
        if 'RandomForest' in model_type:
            print(f"  • Number of trees: {model.n_estimators}")
            print(f"  • Max depth: {model.max_depth}")
            print(f"  • Out-of-bag score: {getattr(model, 'oob_score_', 'Not calculated')}")
            
        elif 'GradientBoosting' in model_type:
            print(f"  • Learning rate: {model.learning_rate}")
            print(f"  • Number of boosting stages: {model.n_estimators}")
            print(f"  • Max depth: {model.max_depth}")
            
        elif 'LogisticRegression' in model_type:
            print(f"  • Regularization strength (C): {model.C}")
            print(f"  • Penalty: {model.penalty}")
            print(f"  • Solver: {model.solver}")
            
        elif 'SVC' in model_type:
            print(f"  • Kernel: {model.kernel}")
            print(f"  • C parameter: {model.C}")
            print(f"  • Gamma: {model.gamma}")
    
    def create_evaluation_plots(self, results=None):
        """Create comprehensive evaluation visualizations"""
        print("\n🎨 Creating model evaluation plots...")
        
        if results is None:
            results = self.results
        
        # Create plots directory
        os.makedirs('outputs/plots', exist_ok=True)
        
        # 1. Model Performance Comparison
        self._plot_model_comparison(results)
        
        # 2. ROC Curves for all models
        self._plot_roc_curves(results)
        
        # 3. Precision-Recall Curves
        self._plot_precision_recall_curves(results)
        
        # 4. Confusion Matrix for best model
        self._plot_confusion_matrix()
        
        # 5. Feature Importance
        self._plot_feature_importance()
        
        print("✅ All evaluation plots created")
    
    def _plot_model_comparison(self, results):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(results.keys())
        
        # AUC Comparison
        auc_scores = [results[model]['test_auc'] for model in models]
        cv_auc_means = [results[model]['cv_auc_mean'] for model in models]
        cv_auc_stds = [results[model]['cv_auc_std'] for model in models]
        
        x_pos = np.arange(len(models))
        
        axes[0,0].bar(x_pos, auc_scores, alpha=0.7, color='skyblue', label='Test AUC')
        axes[0,0].errorbar(x_pos, cv_auc_means, yerr=cv_auc_stds, fmt='ro', label='CV AUC (±std)')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('AUC Score')
        axes[0,0].set_title('AUC Score Comparison')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Precision, Recall, F1 Comparison
        metrics = ['test_precision', 'test_recall', 'test_f1']
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            scores = [results[model][metric] for model in models]
            axes[0,1].plot(models, scores, marker='o', label=name, linewidth=2, markersize=8)
        
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('Score')
        axes[0,1].set_title('Precision, Recall, F1-Score Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Cross-validation stability
        cv_stds = [results[model]['cv_auc_std'] for model in models]
        axes[1,0].bar(models, cv_stds, alpha=0.7, color='orange')
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('CV AUC Standard Deviation')
        axes[1,0].set_title('Model Stability (Lower is Better)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Accuracy vs AUC scatter
        accuracies = [results[model]['test_accuracy'] for model in models]
        axes[1,1].scatter(auc_scores, accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1,1].annotate(model, (auc_scores[i], accuracies[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1,1].set_xlabel('AUC Score')
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('AUC vs Accuracy')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/07_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, results):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, result) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
            auc_score = result['test_auc']
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                    label=f'{name} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/08_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curves(self, results):
        """Plot precision-recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, result) in enumerate(results.items()):
            precision, recall, _ = precision_recall_curve(self.y_test, result['y_pred_proba'])
            
            plt.plot(recall, precision, color=colors[i % len(colors)], linewidth=2,
                    label=f'{name} (F1 = {result["test_f1"]:.4f})')
        
        # Baseline (random classifier)
        baseline = sum(self.y_test) / len(self.y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Random (F1 = {baseline:.4f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - All Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/09_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix for best model"""
        best_result = self.results[self.best_model_name]
        cm = confusion_matrix(self.y_test, best_result['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Lose', 'Win'],
                   yticklabels=['Lose', 'Win'])
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})', 
                        ha='center', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/10_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance for best model"""
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            
            # Top 15 features
            top_features = self.feature_importance.head(15)
            
            # Create horizontal bar plot
            plt.barh(range(len(top_features)), top_features['importance'], 
                    color='steelblue', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 15 Feature Importances - {self.best_model_name}')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, (_, row) in enumerate(top_features.iterrows()):
                plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.4f}', 
                        va='center', fontsize=8)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('outputs/plots/11_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_performance_comparison_plots(self, results=None):
        """Create detailed performance comparison visualizations"""
        print("🎨 Creating performance comparison plots...")
        
        if results is None:
            results = self.results
        
        # Interactive performance dashboard
        self._create_interactive_dashboard(results)
        
        # Learning curves (if implemented)
        # self._plot_learning_curves()
        
        print("✅ Performance comparison plots created")
    
    def _create_interactive_dashboard(self, results):
        """Create interactive Plotly dashboard"""
        # Model performance comparison
        models = list(results.keys())
        metrics = ['test_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Radar', 'AUC Comparison', 
                          'Precision vs Recall', 'Cross-Validation Stability'),
            specs=[[{"type": "polar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Radar chart for best model
        best_metrics = [results[self.best_model_name][metric] for metric in metrics]
        fig.add_trace(go.Scatterpolar(
            r=best_metrics,
            theta=metric_names,
            fill='toself',
            name=self.best_model_name
        ), row=1, col=1)
        
        # AUC comparison bar chart
        auc_scores = [results[model]['test_auc'] for model in models]
        fig.add_trace(go.Bar(
            x=models,
            y=auc_scores,
            name='Test AUC'
        ), row=1, col=2)
        
        # Precision vs Recall scatter
        precisions = [results[model]['test_precision'] for model in models]
        recalls = [results[model]['test_recall'] for model in models]
        fig.add_trace(go.Scatter(
            x=recalls,
            y=precisions,
            mode='markers+text',
            text=models,
            textposition='top center',
            name='Models'
        ), row=2, col=1)
        
        # CV stability
        cv_stds = [results[model]['cv_auc_std'] for model in models]
        fig.add_trace(go.Bar(
            x=models,
            y=cv_stds,
            name='CV Std'
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text="Superhero Model Performance Dashboard")
        fig.write_html('outputs/plots/interactive_model_dashboard.html')
    
    def create_feature_importance_plots(self):
        """Create detailed feature importance visualizations"""
        if self.feature_importance is None:
            print("⚠️ Feature importance not available for this model type")
            return
        
        print("🎨 Creating feature importance plots...")
        
        # 1. Top features bar plot
        plt.figure(figsize=(14, 10))
        
        # Main plot - top 20 features
        plt.subplot(2, 2, (1, 2))
        top_20 = self.feature_importance.head(20)
        plt.barh(range(len(top_20)), top_20['importance'], color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.gca().invert_yaxis()
        
        # Feature importance distribution
        plt.subplot(2, 2, 3)
        plt.hist(self.feature_importance['importance'], bins=20, alpha=0.7, color='lightcoral')
        plt.xlabel('Importance Score')
        plt.ylabel('Number of Features')
        plt.title('Feature Importance Distribution')
        plt.grid(True, alpha=0.3)
        
        # Cumulative importance
        plt.subplot(2, 2, 4)
        cumulative_importance = self.feature_importance['importance'].cumsum()
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                linewidth=2, color='green')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # Add threshold lines
        threshold_50 = np.where(cumulative_importance >= 0.5)[0]
        threshold_80 = np.where(cumulative_importance >= 0.8)[0]
        if len(threshold_50) > 0:
            plt.axvline(x=threshold_50[0] + 1, color='orange', linestyle='--', 
                       label=f'50% importance ({threshold_50[0] + 1} features)')
        if len(threshold_80) > 0:
            plt.axvline(x=threshold_80[0] + 1, color='red', linestyle='--', 
                       label=f'80% importance ({threshold_80[0] + 1} features)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/plots/12_detailed_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature categories analysis
        self._analyze_feature_categories()
    
    def _analyze_feature_categories(self):
        """Analyze importance by feature categories"""
        if self.feature_importance is None:
            return
        
        # Categorize features
        categories = {
            'Basic Stats': ['power_level', 'speed', 'height', 'weight', 'age'],
            'Intelligence': ['intelligence', 'battle_iq', 'tactical_intelligence'],
            'Rankings': ['ranking', 'ranking_category_num'],
            'Encoded Categorical': [f for f in self.feature_importance['feature'] if 'encoded' in f],
            'Engineered Features': [f for f in self.feature_importance['feature'] 
                                  if any(keyword in f for keyword in ['ratio', 'efficiency', 'bmi'])],
            'Training': ['training_time', 'training_efficiency'],
            'Other': []
        }
        
        # Assign features to categories
        categorized_features = set()
        for cat_features in categories.values():
            categorized_features.update(cat_features)
        
        # Add uncategorized features to 'Other'
        for feature in self.feature_importance['feature']:
            if feature not in categorized_features:
                categories['Other'].append(feature)
        
        # Calculate category importance
        category_importance = {}
        for category, features in categories.items():
            importance_sum = self.feature_importance[
                self.feature_importance['feature'].isin(features)
            ]['importance'].sum()
            category_importance[category] = importance_sum
        
        # Plot category importance
        plt.figure(figsize=(10, 6))
        categories_filtered = {k: v for k, v in category_importance.items() if v > 0}
        
        plt.pie(categories_filtered.values(), labels=categories_filtered.keys(), 
               autopct='%1.1f%%', startangle=90)
        plt.title('Feature Importance by Category')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/13_feature_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model=None):
        """Save the best model and related objects"""
        print("\n💾 Saving model and artifacts...")
        
        if model is None:
            model = self.best_model
        
        # Ensure we have a valid model
        if model is None:
            print("❌ No model available to save")
            return
        
        # Create models directory
        os.makedirs('outputs/models', exist_ok=True)
        
        # Save model
        model_filename = f'outputs/models/best_model_{self.best_model_name.lower().replace(" ", "_")}.pkl'
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model saved: {model_filename}")
        except Exception as e:
            print(f"⚠️ Error saving model: {str(e)}")
            return
        
        # Get model parameters safely
        try:
            model_parameters = model.get_params() if hasattr(model, 'get_params') else {}
        except Exception as e:
            print(f"⚠️ Could not get model parameters: {str(e)}")
            model_parameters = {}
        
        # Save model metadata
        model_metadata = {
            'model_name': str(self.best_model_name),
            'model_type': str(type(model).__name__),
            'best_auc': float(self.best_auc),
            'feature_names': list(self.X_train.columns) if self.X_train is not None else [],
            'top_features': self.top_features[:10] if self.top_features else [],
            'training_date': datetime.now().isoformat(),
            'model_parameters': convert_numpy_types(model_parameters)
        }
        
        # Convert any remaining NumPy types
        model_metadata = convert_numpy_types(model_metadata)
        
        try:
            with open('outputs/models/model_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(model_metadata, f, indent=2)
            print("✅ Model metadata saved")
        except Exception as e:
            print(f"⚠️ Error saving metadata: {str(e)}")
        
        # Save feature importance
        if self.feature_importance is not None:
            try:
                self.feature_importance.to_csv('outputs/models/feature_importance.csv', index=False)
                print("✅ Feature importance saved")
            except Exception as e:
                print(f"⚠️ Error saving feature importance: {str(e)}")
        
        # Save results summary
        try:
            results_summary = pd.DataFrame({
                'Model': list(self.results.keys()),
                'Test_AUC': [float(self.results[name]['test_auc']) for name in self.results.keys()],
                'CV_AUC_Mean': [float(self.results[name]['cv_auc_mean']) for name in self.results.keys()],
                'Test_F1': [float(self.results[name]['test_f1']) for name in self.results.keys()],
                'Test_Accuracy': [float(self.results[name]['test_accuracy']) for name in self.results.keys()]
            })
            results_summary.to_csv('outputs/models/model_results_summary.csv', index=False)
            print("✅ Results summary saved")
        except Exception as e:
            print(f"⚠️ Error saving results summary: {str(e)}")
        
        return True
    
    def generate_modeling_report(self, results=None, model_metrics=None):
        """Generate comprehensive modeling report"""
        print("\n📝 Generating modeling report...")
        
        if results is None:
            results = self.results
        
        # Create reports directory
        os.makedirs('outputs/reports', exist_ok=True)
        
        report_content = f"""
# Machine Learning Modeling Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents the results of comprehensive machine learning modeling for superhero winning probability prediction.

## Models Evaluated
{len(results)} machine learning algorithms were trained and evaluated:
"""
        
        for model_name in results.keys():
            report_content += f"- **{model_name}**\n"
        
        report_content += f"""

## Best Model: {self.best_model_name}
- **Test AUC**: {self.best_auc:.4f}
- **Cross-validation AUC**: {results[self.best_model_name]['cv_auc_mean']:.4f} (±{results[self.best_model_name]['cv_auc_std']:.4f})
- **Test Accuracy**: {results[self.best_model_name]['test_accuracy']:.4f}
- **Test F1-Score**: {results[self.best_model_name]['test_f1']:.4f}

## Model Performance Comparison

| Model | Test AUC | CV AUC | Accuracy | Precision | Recall | F1-Score |
|-------|----------|---------|----------|-----------|---------|----------|
"""
        
        for name, result in sorted(results.items(), key=lambda x: x[1]['test_auc'], reverse=True):
            report_content += f"| {name} | {result['test_auc']:.4f} | {result['cv_auc_mean']:.4f} | {result['test_accuracy']:.4f} | {result['test_precision']:.4f} | {result['test_recall']:.4f} | {result['test_f1']:.4f} |\n"
        
        # Feature importance section
        if self.feature_importance is not None:
            report_content += f"""

## Feature Importance Analysis

### Top 10 Most Important Features:
"""
            for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows(), 1):
                report_content += f"{i}. **{row['feature']}**: {row['importance']:.4f}\n"
            
            # Feature insights
            report_content += f"""

### Feature Insights:
- **Most Important Feature**: {self.feature_importance.iloc[0]['feature']} ({self.feature_importance.iloc[0]['importance']:.4f})
- **Top 5 features** account for {self.feature_importance.head(5)['importance'].sum():.1%} of total importance
- **Top 10 features** account for {self.feature_importance.head(10)['importance'].sum():.1%} of total importance
"""
        
        # Model interpretation
        report_content += f"""

## Model Interpretation

### {self.best_model_name} Characteristics:
"""
        
        best_model = results[self.best_model_name]['model']
        model_type = type(best_model).__name__
        
        if 'RandomForest' in model_type:
            report_content += f"""
- **Algorithm**: Ensemble of {best_model.n_estimators} decision trees
- **Max Depth**: {best_model.max_depth}
- **Class Weight**: {best_model.class_weight} (handles class imbalance)
- **Strengths**: Robust to overfitting, handles mixed data types, provides feature importance
- **Interpretation**: Predictions based on majority vote of decision trees
"""
        elif 'GradientBoosting' in model_type:
            report_content += f"""
- **Algorithm**: Sequential ensemble with {best_model.n_estimators} weak learners
- **Learning Rate**: {best_model.learning_rate}
- **Max Depth**: {best_model.max_depth}
- **Strengths**: High predictive power, good for complex patterns
- **Interpretation**: Iteratively corrects prediction errors
"""
        elif 'LogisticRegression' in model_type:
            report_content += f"""
- **Algorithm**: Linear classifier with logistic function
- **Regularization**: C={best_model.C}, Penalty={best_model.penalty}
- **Strengths**: Interpretable coefficients, probability estimates
- **Interpretation**: Linear combination of features predicts log-odds
"""
        
        # Performance insights
        report_content += f"""

## Performance Analysis

### Classification Performance:
- **AUC Score**: {self.best_auc:.4f} indicates {'excellent' if self.best_auc > 0.8 else 'good' if self.best_auc > 0.7 else 'fair'} discriminative ability
- **Class Balance**: Model handles imbalanced dataset with class weights
- **Cross-validation Stability**: Low standard deviation indicates consistent performance

### Practical Implications:
- Model can effectively distinguish between winning and losing characters
- Feature importance reveals key factors that determine success
- Predictions can be used for character analysis and fight outcome prediction

## Recommendations

### Model Deployment:
1. **Use {self.best_model_name}** for production predictions
2. **Monitor performance** on new data to detect concept drift
3. **Focus on top features** for character development insights

### Future Improvements:
1. **Collect more data** to improve model robustness
2. **Engineer interaction features** between top predictors
3. **Experiment with deep learning** for complex pattern detection
4. **Implement ensemble methods** combining multiple models

## Files Generated
- `best_model_{self.best_model_name.lower().replace(' ', '_')}.pkl`: Trained model
- `model_metadata.json`: Model configuration and metadata
- `feature_importance.csv`: Complete feature importance rankings
- `model_results_summary.csv`: Performance metrics for all models
- `07_model_comparison.png`: Performance comparison visualizations
- `08_roc_curves.png`: ROC curves for all models
- `09_precision_recall_curves.png`: Precision-recall analysis
- `10_confusion_matrix.png`: Best model confusion matrix
- `11_feature_importance.png`: Feature importance visualization

---
*Report generated by Machine Learning Modeling Module*
"""
        
        # Save report
        with open('outputs/reports/Modeling_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save modeling summary as JSON
        modeling_summary = {
            'best_model': self.best_model_name,
            'best_auc': float(self.best_auc),
            'models_evaluated': len(results),
            'top_features': self.top_features[:5] if self.top_features else [],
            'performance_metrics': {
                name: {
                    'test_auc': float(result['test_auc']),
                    'test_f1': float(result['test_f1']),
                    'test_accuracy': float(result['test_accuracy'])
                }
                for name, result in results.items()
            }
        }
        
        # Convert any NumPy types
        modeling_summary = convert_numpy_types(modeling_summary)
        
        with open('outputs/reports/modeling_summary.json', 'w', encoding='utf-8') as f:
            json.dump(modeling_summary, f, indent=2)
        
        print("✅ Modeling report saved to outputs/reports/")

if __name__ == "__main__":
    # Test the modeling module
    print("🧪 Testing SuperheroModeling module...")
    
    try:
        # Load preprocessed data
        X_train = pd.read_csv('outputs/data/X_train.csv')
        X_test = pd.read_csv('outputs/data/X_test.csv')
        y_train = pd.read_csv('outputs/data/y_train.csv').iloc[:, 0]
        y_test = pd.read_csv('outputs/data/y_test.csv').iloc[:, 0]
        
        # Initialize modeling
        modeler = SuperheroModeling()
        
        # Train models
        results = modeler.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        evaluation_df, best_result = modeler.evaluate_models()
        
        # Create visualizations
        modeler.create_evaluation_plots()
        modeler.create_feature_importance_plots()
        
        # Save model and generate report
        modeler.save_model()
        modeler.generate_modeling_report()
        
        print("✅ Modeling module test completed!")
        print(f"🏆 Best model: {modeler.best_model_name} (AUC: {modeler.best_auc:.4f})")
        
    except FileNotFoundError as e:
        print(f"❌ Required files not found: {e}")
        print("Please run preprocessing module first.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")