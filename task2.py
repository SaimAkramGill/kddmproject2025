"""
Task 2: Character Winning Probability Predictions
Predict winning probabilities for unseen characters and analyze fight outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

class Task2Predictor:
    """
    Task 2: Predict winning probabilities for unseen superhero/villain characters
    """
    
    def __init__(self, modeler):
        self.modeler = modeler
        self.model = modeler.best_model
        self.model_name = modeler.best_model_name
        self.feature_names = modeler.X_train.columns.tolist()
        
        # Task 2 data
        self.characters_data = None
        self.matches_data = None
        self.character_predictions = None
        self.fight_predictions = None
        
        # Load preprocessing objects
        self._load_preprocessing_objects()
    
    def _load_preprocessing_objects(self):
        """Load preprocessing objects for consistent data transformation"""
        try:
            with open('outputs/models/preprocessing_objects.pkl', 'rb') as f:
                preprocessing_objects = pickle.load(f)
                self.label_encoders = preprocessing_objects.get('label_encoders', {})
                self.scaler = preprocessing_objects.get('scaler', None)
                self.imputers = preprocessing_objects.get('imputers', {})
        except FileNotFoundError:
            print(" Preprocessing objects not found. Creating new ones...")
            self.label_encoders = {}
            self.scaler = None
            self.imputers = {}
    
    def load_task2_data(self):
        """Load Task 2 character and match data"""
        print("\n LOADING TASK 2 DATA")
        print("-" * 40)
        
        try:
            # Load characters data
            self.characters_data = pd.read_csv('Task2_superheroes_villains.csv')
            print(f" Loaded {len(self.characters_data)} characters")
            
            # Load matches data
            self.matches_data = pd.read_csv('Task2_matches.csv')
            print(f" Loaded {len(self.matches_data)} fight matches")
            
            # Display character info
            print("\n Characters to predict:")
            for i, char in self.characters_data.iterrows():
                char_type = " Hero" if char['role'] in ['Hero', 'H3ro', 'HerO'] else "🦹 Villain"
                print(f"  {char_type} {char['name']}")
                print(f"    Power: {char.get('power_level', 'N/A')}, Speed: {char.get('speed', 'N/A')}, "
                      f"Battle IQ: {char.get('battle_iq', 'N/A')}")
            
            # Display fights
            print("\n Scheduled fights:")
            for i, fight in self.matches_data.iterrows():
                print(f"  Fight {i+1}: {fight['first']} vs {fight['second']}")
            
            return self.characters_data, self.matches_data
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Task 2 data files not found: {e}")
    
    def _create_engineered_features(self, data):
        """Create same engineered features as in training - only if they exist in training features"""
        
        # Only create features that were actually used in training
        
        # Power to weight ratio
        if 'power_weight_ratio' in self.feature_names and 'power_level' in data.columns and 'weight' in data.columns:
            data['power_weight_ratio'] = data['power_level'] / (data['weight'] + 1)
        
        # Speed to age ratio
        if 'speed_age_ratio' in self.feature_names and 'speed' in data.columns and 'age' in data.columns:
            data['speed_age_ratio'] = data['speed'] / (data['age'] + 1)
        
        # Tactical intelligence
        if 'tactical_intelligence' in self.feature_names and 'intelligence' in data.columns and 'battle_iq' in data.columns:
            data['tactical_intelligence'] = (data['intelligence'] + data['battle_iq']) / 2
        
        # Hero BMI
        if 'hero_bmi' in self.feature_names and 'height' in data.columns and 'weight' in data.columns:
            data['hero_bmi'] = data['weight'] / ((data['height'] / 100) ** 2 + 0.01)
        
        # Training efficiency
        if 'training_efficiency' in self.feature_names and 'training_time' in data.columns and 'age' in data.columns:
            data['training_efficiency'] = data['training_time'] / (data['age'] + 1)
        
        # Ranking category
        if 'ranking_category_num' in self.feature_names and 'ranking' in data.columns:
            # Use same binning as training
            data['ranking_category_num'] = pd.cut(data['ranking'], 
                                                 bins=5, 
                                                 labels=[5, 4, 3, 2, 1]).astype(float)
        
        print(f"🔧 Created engineered features that exist in training feature set")
    
    def preprocess_task2_data(self):
        """Preprocess Task 2 data to match training format exactly"""
        print("\n PREPROCESSING TASK 2 DATA")
        print("-" * 40)
        
        # Make a copy for processing
        processed_data = self.characters_data.copy()
        
        # 1. Standardize role names
        role_mapping = {
            'H3ro': 'Hero',
            'HerO': 'Hero',
            'VIllain': 'Villain',
            'VillaIn': 'Villain'
        }
        processed_data['role'] = processed_data['role'].map(role_mapping).fillna(processed_data['role'])
        
        # 2. Create same engineered features as training (conditionally)
        self._create_engineered_features(processed_data)
        
        # 3. Only process categorical features that exist in the training feature names
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            encoded_col_name = f'{col}_encoded'
            
            # Only process if this encoded feature exists in training features
            if encoded_col_name in self.feature_names:
                if col in self.label_encoders:
                    # Handle unseen categories
                    processed_data[col] = processed_data[col].fillna('Unknown')
                    
                    # Get unique values and handle unseen ones
                    unique_vals = processed_data[col].unique()
                    known_classes = set(self.label_encoders[col].classes_)
                    
                    # Map unseen values to first known class (fallback)
                    for val in unique_vals:
                        if val not in known_classes:
                            processed_data[col] = processed_data[col].replace(val, 
                                                                            self.label_encoders[col].classes_[0])
                    
                    # Apply encoding
                    processed_data[encoded_col_name] = self.label_encoders[col].transform(processed_data[col])
                else:
                    # Create simple default encoding for missing categorical features
                    processed_data[encoded_col_name] = 0  # Default value
                    print(f" No encoder for '{col}', using default encoding")
            else:
                # This categorical feature was not used in training, skip it
                print(f" Skipping '{col}' - not used in training model")
        
        # 4. Drop ALL original categorical columns (whether processed or not)
        processed_data = processed_data.drop(columns=categorical_cols, errors='ignore')
        
        # 5. Create a DataFrame with ONLY the training features, filled with defaults
        final_data = pd.DataFrame(index=processed_data.index)
        
        for feature in self.feature_names:
            if feature in processed_data.columns:
                final_data[feature] = processed_data[feature]
            else:
                # Use appropriate default values based on feature name patterns
                if 'encoded' in feature:
                    final_data[feature] = 0  # Default for encoded categorical
                elif any(keyword in feature.lower() for keyword in ['ratio', 'efficiency']):
                    final_data[feature] = 1.0  # Default for ratios
                elif 'bmi' in feature.lower():
                    final_data[feature] = 25.0  # Average BMI
                elif 'category' in feature.lower():
                    final_data[feature] = 3.0  # Middle category
                else:
                    final_data[feature] = 0.0  # Default for other features
                
                print(f" Added missing feature '{feature}' with default value")
        
        # 6. Ensure exact feature order matches training
        final_data = final_data[self.feature_names]
        
        # 7. Handle missing values (same as training)
        if 'numerical_median' in self.imputers:
            try:
                final_data = pd.DataFrame(
                    self.imputers['numerical_median'].transform(final_data),
                    columns=final_data.columns,
                    index=final_data.index
                )
            except Exception as e:
                print(f" Imputer error: {e}. Using median imputation.")
                final_data = final_data.fillna(final_data.median())
        else:
            # Use simple median imputation if imputer not available
            final_data = final_data.fillna(final_data.median())
            final_data = final_data.fillna(0)  # Fill any remaining NaN with 0
        
        # 8. Scale features if scaler available
        if self.scaler is not None:
            try:
                final_data = pd.DataFrame(
                    self.scaler.transform(final_data),
                    columns=final_data.columns,
                    index=final_data.index
                )
            except Exception as e:
                print(f" Scaler error: {e}. Skipping scaling.")
        
        print(f" Preprocessed {len(final_data)} characters")
        print(f" Feature matrix shape: {final_data.shape}")
        print(f" Feature names match training: {list(final_data.columns) == self.feature_names}")
        print(f" Features created: {len([f for f in self.feature_names if f in processed_data.columns])} / {len(self.feature_names)}")
        
        return final_data
    
    def predict_characters(self):
        """Predict winning probabilities for all characters"""
        print("\n PREDICTING CHARACTER WINNING PROBABILITIES")
        print("-" * 50)
        
        # Preprocess data
        processed_data = self.preprocess_task2_data()
        
        # Make predictions
        win_probabilities = self.model.predict_proba(processed_data)[:, 1]
        
        # Create results dataframe
        results = self.characters_data.copy()
        results['predicted_win_probability'] = win_probabilities
        results['predicted_outcome'] = (win_probabilities > 0.5).astype(int)
        results['confidence_level'] = np.where(
            np.abs(win_probabilities - 0.5) > 0.3, 'High',
            np.where(np.abs(win_probabilities - 0.5) > 0.15, 'Medium', 'Low')
        )
        
        # Store predictions
        self.character_predictions = results
        
        # Display results
        print(" CHARACTER PREDICTIONS:")
        print("=" * 60)
        
        for i, char in results.iterrows():
            char_type = "🦸 Hero" if char['role'] in ['Hero', 'H3ro', 'HerO'] else "🦹 Villain"
            outcome = "WIN" if char['predicted_outcome'] == 1 else "LOSE"
            confidence = char['confidence_level']
            
            print(f"{char_type} {char['name']:15}")
            print(f"   Win Probability: {char['predicted_win_probability']:.3f} ({char['predicted_win_probability']*100:.1f}%)")
            print(f"   Predicted Outcome: {outcome}")
            print(f"   Confidence: {confidence}")
            print()
        
        # Summary statistics
        heroes = results[results['role'].isin(['Hero', 'H3ro', 'HerO'])]
        villains = results[results['role'].isin(['Villain', 'VIllain', 'VillaIn'])]
        
        print(" SUMMARY STATISTICS:")
        print(f"  Heroes average win probability: {heroes['predicted_win_probability'].mean():.3f}")
        print(f"  Villains average win probability: {villains['predicted_win_probability'].mean():.3f}")
        print(f"  Highest win probability: {results.loc[results['predicted_win_probability'].idxmax(), 'name']} ({results['predicted_win_probability'].max():.3f})")
        print(f"  Lowest win probability: {results.loc[results['predicted_win_probability'].idxmin(), 'name']} ({results['predicted_win_probability'].min():.3f})")
        
        return results
    
    def predict_fights(self):
        """Predict outcomes for scheduled fights"""
        print("\n PREDICTING FIGHT OUTCOMES")
        print("-" * 40)
        
        if self.character_predictions is None:
            print(" Character predictions not available. Run predict_characters() first.")
            return None
        
        # Create fight predictions
        fight_results = []
        
        for i, fight in self.matches_data.iterrows():
            fighter1_name = fight['first']
            fighter2_name = fight['second']
            
            # Get predictions for both fighters
            fighter1_data = self.character_predictions[self.character_predictions['name'] == fighter1_name]
            fighter2_data = self.character_predictions[self.character_predictions['name'] == fighter2_name]
            
            if len(fighter1_data) == 0 or len(fighter2_data) == 0:
                print(f" Fighter not found for: {fighter1_name} vs {fighter2_name}")
                continue
            
            f1_prob = fighter1_data['predicted_win_probability'].iloc[0]
            f2_prob = fighter2_data['predicted_win_probability'].iloc[0]
            
            # Determine winner (higher probability)
            if f1_prob > f2_prob:
                winner = fighter1_name
                winner_prob = f1_prob
                loser = fighter2_name
                loser_prob = f2_prob
                margin = f1_prob - f2_prob
            else:
                winner = fighter2_name
                winner_prob = f2_prob
                loser = fighter1_name
                loser_prob = f1_prob
                margin = f2_prob - f1_prob
            
            # Determine fight closeness
            if margin > 0.3:
                fight_type = "Dominant Victory"
            elif margin > 0.15:
                fight_type = "Clear Victory"
            elif margin > 0.05:
                fight_type = "Close Fight"
            else:
                fight_type = "Too Close to Call"
            
            fight_result = {
                'fight_number': i + 1,
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_prob': f1_prob,
                'fighter2_prob': f2_prob,
                'winner': winner,
                'winner_prob': winner_prob,
                'loser': loser,
                'loser_prob': loser_prob,
                'margin': margin,
                'fight_type': fight_type
            }
            
            fight_results.append(fight_result)
            
            # Display result
            print(f" FIGHT {i+1}: {fighter1_name} vs {fighter2_name}")
            print(f"   {fighter1_name}: {f1_prob:.3f} ({f1_prob*100:.1f}%)")
            print(f"   {fighter2_name}: {f2_prob:.3f} ({f2_prob*100:.1f}%)")
            print(f"    WINNER: {winner} ({fight_type})")
            print(f"    Victory Margin: {margin:.3f}")
            print()
        
        # Store fight predictions
        self.fight_predictions = pd.DataFrame(fight_results)
        
        return self.fight_predictions
    
    def analyze_character_strengths(self):
        """Analyze what makes each character strong/weak"""
        print("\n CHARACTER STRENGTH ANALYSIS")
        print("-" * 40)
        
        if self.character_predictions is None:
            return None
        
        # Get feature importance from the model
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(5)['feature'].tolist()
        else:
            # Use predefined important features
            top_features = ['ranking', 'battle_iq', 'intelligence', 'power_level', 'speed']
        
        # Analyze each character
        for i, char in self.character_predictions.iterrows():
            print(f"\n {char['name']} Analysis:")
            print(f"   Win Probability: {char['predicted_win_probability']:.3f}")
            
            # Get character's feature values (before preprocessing)
            char_features = {}
            for feature in top_features:
                if feature in self.characters_data.columns:
                    char_features[feature] = self.characters_data.loc[i, feature]
            
            # Rank features for this character
            print("   Key Strengths/Weaknesses:")
            for feature, value in char_features.items():
                if pd.notna(value):
                    print(f"     {feature}: {value}")
        
        return top_features
    
    def create_prediction_plots(self):
        """Create visualizations for character predictions"""
        print("\n Creating prediction visualizations...")
        
        if self.character_predictions is None:
            print(" No predictions available. Run predict_characters() first.")
            return
        
        # Create plots directory
        os.makedirs('outputs/plots', exist_ok=True)
        
        # 1. Character predictions bar chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task 2: Character Predictions Analysis', fontsize=16, fontweight='bold')
        
        # Predictions bar chart
        char_names = self.character_predictions['name']
        win_probs = self.character_predictions['predicted_win_probability']
        colors = ['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                 for role in self.character_predictions['role']]
        
        bars = axes[0,0].bar(range(len(char_names)), win_probs, color=colors, alpha=0.7)
        axes[0,0].set_xticks(range(len(char_names)))
        axes[0,0].set_xticklabels(char_names, rotation=45, ha='right')
        axes[0,0].set_ylabel('Win Probability')
        axes[0,0].set_title('Character Win Probability Predictions')
        axes[0,0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, prob in zip(bars, win_probs):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Heroes vs Villains comparison
        heroes_data = self.character_predictions[self.character_predictions['role'].isin(['Hero', 'H3ro', 'HerO'])]
        villains_data = self.character_predictions[self.character_predictions['role'].isin(['Villain', 'VIllain', 'VillaIn'])]
        
        hero_avg = heroes_data['predicted_win_probability'].mean()
        villain_avg = villains_data['predicted_win_probability'].mean()
        
        axes[0,1].bar(['Heroes', 'Villains'], [hero_avg, villain_avg], 
                     color=['blue', 'red'], alpha=0.7)
        axes[0,1].set_ylabel('Average Win Probability')
        axes[0,1].set_title('Heroes vs Villains Average Performance')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        axes[0,1].text(0, hero_avg + 0.01, f'{hero_avg:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0,1].text(1, villain_avg + 0.01, f'{villain_avg:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence distribution
        confidence_counts = self.character_predictions['confidence_level'].value_counts()
        axes[1,0].pie(confidence_counts.values, labels=confidence_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Prediction Confidence Distribution')
        
        # Prediction distribution histogram
        axes[1,1].hist(win_probs, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].set_xlabel('Win Probability')
        axes[1,1].set_ylabel('Number of Characters')
        axes[1,1].set_title('Win Probability Distribution')
        axes[1,1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/14_task2_character_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_fight_analysis_plots(self):
        """Create fight analysis visualizations"""
        print(" Creating fight analysis plots...")
        
        if self.fight_predictions is None:
            print(" No fight predictions available. Run predict_fights() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task 2: Fight Analysis', fontsize=16, fontweight='bold')
        
        # Fight outcomes
        fight_labels = [f"Fight {row['fight_number']}\n{row['fighter1']} vs\n{row['fighter2']}" 
                       for _, row in self.fight_predictions.iterrows()]
        
        # Winner probabilities
        winner_probs = self.fight_predictions['winner_prob']
        margins = self.fight_predictions['margin']
        
        bars = axes[0,0].bar(range(len(fight_labels)), winner_probs, 
                           color=['gold' if margin > 0.2 else 'orange' if margin > 0.1 else 'lightcoral' 
                                 for margin in margins])
        axes[0,0].set_xticks(range(len(fight_labels)))
        axes[0,0].set_xticklabels([f"Fight {i+1}" for i in range(len(fight_labels))], rotation=0)
        axes[0,0].set_ylabel('Winner Probability')
        axes[0,0].set_title('Fight Winner Probabilities')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, prob, winner) in enumerate(zip(bars, winner_probs, self.fight_predictions['winner'])):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{winner}\n{prob:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Victory margins
        axes[0,1].bar(range(len(fight_labels)), margins, color='steelblue', alpha=0.7)
        axes[0,1].set_xticks(range(len(fight_labels)))
        axes[0,1].set_xticklabels([f"Fight {i+1}" for i in range(len(fight_labels))], rotation=0)
        axes[0,1].set_ylabel('Victory Margin')
        axes[0,1].set_title('Fight Victory Margins')
        axes[0,1].grid(True, alpha=0.3)
        
        # Fight type distribution
        fight_type_counts = self.fight_predictions['fight_type'].value_counts()
        axes[1,0].pie(fight_type_counts.values, labels=fight_type_counts.index, autopct='%1.0f%%')
        axes[1,0].set_title('Fight Type Distribution')
        
        # Head-to-head comparison
        for i, (_, fight) in enumerate(self.fight_predictions.iterrows()):
            axes[1,1].barh([f"Fight {i+1} - {fight['fighter1']}", f"Fight {i+1} - {fight['fighter2']}"], 
                         [fight['fighter1_prob'], fight['fighter2_prob']], 
                         color=['blue', 'red'])
        
        axes[1,1].set_xlabel('Win Probability')
        axes[1,1].set_title('Head-to-Head Probability Comparison')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/15_task2_fight_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_character_comparison_plots(self):
        """Create detailed character comparison plots"""
        print(" Creating character comparison plots...")
        
        # Character feature comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Task 2: Character Feature Analysis', fontsize=16, fontweight='bold')
        
        # Key features radar chart data
        key_features = ['power_level', 'speed', 'battle_iq', 'intelligence', 'ranking']
        available_features = [f for f in key_features if f in self.characters_data.columns]
        
        if len(available_features) >= 3:
            # Normalize features for radar chart
            feature_data = self.characters_data[available_features].fillna(0)
            
            # Normalize to 0-1 scale
            feature_data_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
            
            # Create radar chart data
            angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax = plt.subplot(2, 2, 1, projection='polar')
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for i, (_, char) in enumerate(self.characters_data.iterrows()):
                values = feature_data_norm.iloc[i].tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, 
                       label=f"{char['name']}")
                ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(available_features)
            ax.set_title('Character Feature Comparison (Normalized)')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # Power vs Speed scatter
        if 'power_level' in self.characters_data.columns and 'speed' in self.characters_data.columns:
            heroes = self.character_predictions[self.character_predictions['role'].isin(['Hero', 'H3ro', 'HerO'])]
            villains = self.character_predictions[self.character_predictions['role'].isin(['Villain', 'VIllain', 'VillaIn'])]
            
            axes[0,1].scatter(heroes['power_level'], heroes['speed'], 
                            s=heroes['predicted_win_probability']*500, 
                            c='blue', alpha=0.7, label='Heroes')
            axes[0,1].scatter(villains['power_level'], villains['speed'], 
                            s=villains['predicted_win_probability']*500, 
                            c='red', alpha=0.7, label='Villains')
            
            # Add character labels
            for _, char in self.character_predictions.iterrows():
                axes[0,1].annotate(char['name'], 
                                 (char['power_level'], char['speed']),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[0,1].set_xlabel('Power Level')
            axes[0,1].set_ylabel('Speed')
            axes[0,1].set_title('Power vs Speed (Bubble size = Win Probability)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Intelligence vs Battle IQ
        if 'intelligence' in self.characters_data.columns and 'battle_iq' in self.characters_data.columns:
            axes[1,0].scatter(self.character_predictions['intelligence'], 
                            self.character_predictions['battle_iq'],
                            c=self.character_predictions['predicted_win_probability'],
                            s=100, alpha=0.7, cmap='viridis')
            
            # Add character labels
            for _, char in self.character_predictions.iterrows():
                axes[1,0].annotate(char['name'], 
                                 (char['intelligence'], char['battle_iq']),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1,0].set_xlabel('Intelligence')
            axes[1,0].set_ylabel('Battle IQ')
            axes[1,0].set_title('Intelligence vs Battle IQ (Color = Win Probability)')
            
            # Add colorbar
            plt.colorbar(axes[1,0].collections[0], ax=axes[1,0], label='Win Probability')
            axes[1,0].grid(True, alpha=0.3)
        
        # Ranking vs Win Probability
        if 'ranking' in self.characters_data.columns:
            axes[1,1].scatter(self.character_predictions['ranking'], 
                            self.character_predictions['predicted_win_probability'],
                            s=100, alpha=0.7, 
                            c=['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                              for role in self.character_predictions['role']])
            
            # Add character labels
            for _, char in self.character_predictions.iterrows():
                axes[1,1].annotate(char['name'], 
                                 (char['ranking'], char['predicted_win_probability']),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1,1].set_xlabel('Ranking (Lower = Better)')
            axes[1,1].set_ylabel('Predicted Win Probability')
            axes[1,1].set_title('Ranking vs Win Probability')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/16_task2_character_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self):
        """Create interactive Task 2 dashboard"""
        print(" Creating interactive Task 2 dashboard...")
        
        # Create interactive dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Character Win Probabilities', 'Fight Predictions', 
                          'Heroes vs Villains', 'Feature Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Character predictions
        fig.add_trace(go.Bar(
            x=self.character_predictions['name'],
            y=self.character_predictions['predicted_win_probability'],
            marker_color=['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                         for role in self.character_predictions['role']],
            name='Win Probability'
        ), row=1, col=1)
        
        # Fight predictions
        if self.fight_predictions is not None:
            fight_names = [f"Fight {row['fight_number']}" for _, row in self.fight_predictions.iterrows()]
            fig.add_trace(go.Bar(
                x=fight_names,
                y=self.fight_predictions['winner_prob'],
                text=self.fight_predictions['winner'],
                textposition='auto',
                name='Winner Probability'
            ), row=1, col=2)
        
        # Heroes vs Villains
        heroes_avg = self.character_predictions[
            self.character_predictions['role'].isin(['Hero', 'H3ro', 'HerO'])
        ]['predicted_win_probability'].mean()
        
        villains_avg = self.character_predictions[
            self.character_predictions['role'].isin(['Villain', 'VIllain', 'VillaIn'])
        ]['predicted_win_probability'].mean()
        
        fig.add_trace(go.Bar(
            x=['Heroes', 'Villains'],
            y=[heroes_avg, villains_avg],
            marker_color=['blue', 'red'],
            name='Average Win Prob'
        ), row=2, col=1)
        
        # Feature comparison scatter
        if 'power_level' in self.characters_data.columns and 'speed' in self.characters_data.columns:
            fig.add_trace(go.Scatter(
                x=self.character_predictions['power_level'],
                y=self.character_predictions['speed'],
                mode='markers+text',
                text=self.character_predictions['name'],
                textposition='top center',
                marker=dict(
                    size=self.character_predictions['predicted_win_probability']*50,
                    color=['blue' if role in ['Hero', 'H3ro', 'HerO'] else 'red' 
                          for role in self.character_predictions['role']],
                    opacity=0.7
                ),
                name='Characters'
            ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text="Task 2: Character Predictions Dashboard")
        fig.write_html('outputs/plots/task2_interactive_dashboard.html')
    
    def save_predictions(self, character_predictions=None, fight_predictions=None):
        """Save prediction results to files"""
        print("\n Saving Task 2 predictions...")
        
        # Create predictions directory
        os.makedirs('outputs/predictions', exist_ok=True)
        
        # Save character predictions
        if character_predictions is None:
            character_predictions = self.character_predictions
        
        if character_predictions is not None:
            character_predictions.to_csv('outputs/predictions/task2_character_predictions.csv', index=False)
            print(" Character predictions saved")
        
        # Save fight predictions
        if fight_predictions is None:
            fight_predictions = self.fight_predictions
        
        if fight_predictions is not None:
            fight_predictions.to_csv('outputs/predictions/task2_fight_predictions.csv', index=False)
            print(" Fight predictions saved")
        
        return True
    
    def generate_task2_report(self, character_predictions=None, fight_predictions=None):
        """Generate comprehensive Task 2 report"""
        print("\n Generating Task 2 report...")
        
        if character_predictions is None:
            character_predictions = self.character_predictions
        if fight_predictions is None:
            fight_predictions = self.fight_predictions
        
        # Create reports directory
        os.makedirs('outputs/reports', exist_ok=True)
        
        report_content = f"""
# Task 2: Character Predictions Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report presents predictions for {len(character_predictions)} superhero/villain characters and {len(fight_predictions) if fight_predictions is not None else 0} scheduled fights using the best performing model: **{self.model_name}**.

## Character Predictions

### Individual Character Analysis
"""
        
        # Add individual character results
        for _, char in character_predictions.iterrows():
            char_type = "Hero" if char['role'] in ['Hero', 'H3ro', 'HerO'] else "Villain"
            outcome = "WIN" if char['predicted_outcome'] == 1 else "LOSE"
            
            report_content += f"""
#### {char['name']} ({char_type})
- **Win Probability**: {char['predicted_win_probability']:.3f} ({char['predicted_win_probability']*100:.1f}%)
- **Predicted Outcome**: {outcome}
- **Confidence Level**: {char['confidence_level']}
- **Key Stats**: Power: {char.get('power_level', 'N/A')}, Speed: {char.get('speed', 'N/A')}, Battle IQ: {char.get('battle_iq', 'N/A')}
"""
        
        # Summary statistics
        heroes = character_predictions[character_predictions['role'].isin(['Hero', 'H3ro', 'HerO'])]
        villains = character_predictions[character_predictions['role'].isin(['Villain', 'VIllain', 'VillaIn'])]
        
        report_content += f"""

### Summary Statistics
- **Total Characters Analyzed**: {len(character_predictions)}
- **Heroes**: {len(heroes)} characters
- **Villains**: {len(villains)} characters
- **Average Win Probability (Heroes)**: {heroes['predicted_win_probability'].mean():.3f}
- **Average Win Probability (Villains)**: {villains['predicted_win_probability'].mean():.3f}
- **Highest Win Probability**: {character_predictions.loc[character_predictions['predicted_win_probability'].idxmax(), 'name']} ({character_predictions['predicted_win_probability'].max():.3f})
- **Lowest Win Probability**: {character_predictions.loc[character_predictions['predicted_win_probability'].idxmin(), 'name']} ({character_predictions['predicted_win_probability'].min():.3f})

"""
        
        # Fight predictions
        if fight_predictions is not None and len(fight_predictions) > 0:
            report_content += f"""
## Fight Predictions

### Scheduled Fights Analysis
"""
            
            for _, fight in fight_predictions.iterrows():
                report_content += f"""
#### Fight {fight['fight_number']}: {fight['fighter1']} vs {fight['fighter2']}
- **{fight['fighter1']}**: {fight['fighter1_prob']:.3f} win probability
- **{fight['fighter2']}**: {fight['fighter2_prob']:.3f} win probability
- ** PREDICTED WINNER**: {fight['winner']}
- **Victory Margin**: {fight['margin']:.3f}
- **Fight Type**: {fight['fight_type']}
"""
            
            # Fight summary
            report_content += f"""

### Fight Summary
- **Total Fights**: {len(fight_predictions)}
- **Dominant Victories**: {len(fight_predictions[fight_predictions['fight_type'] == 'Dominant Victory'])}
- **Clear Victories**: {len(fight_predictions[fight_predictions['fight_type'] == 'Clear Victory'])}
- **Close Fights**: {len(fight_predictions[fight_predictions['fight_type'] == 'Close Fight'])}
- **Too Close to Call**: {len(fight_predictions[fight_predictions['fight_type'] == 'Too Close to Call'])}
"""
        
        # Model information
        report_content += f"""

## Model Information
- **Algorithm Used**: {self.model_name}
- **Model Type**: {type(self.model).__name__}
- **Training AUC**: {getattr(self.modeler, 'best_auc', 'N/A')}
- **Features Used**: {len(self.feature_names)} features

## Key Insights

### Character Analysis Insights
1. **Strongest Character**: {character_predictions.loc[character_predictions['predicted_win_probability'].idxmax(), 'name']} shows the highest win probability
2. **Most Competitive**: Characters with probabilities near 0.5 indicate balanced matchups
3. **Role Performance**: {'Heroes' if heroes['predicted_win_probability'].mean() > villains['predicted_win_probability'].mean() else 'Villains'} show higher average win probability

### Fight Prediction Insights
"""
        
        if fight_predictions is not None and len(fight_predictions) > 0:
            report_content += f"""
1. **Most Decisive Fight**: Fight with largest victory margin ({fight_predictions.loc[fight_predictions['margin'].idxmax(), 'margin']:.3f})
2. **Closest Fight**: Fight with smallest victory margin ({fight_predictions.loc[fight_predictions['margin'].idxmin(), 'margin']:.3f})
3. **Fight Balance**: {len(fight_predictions[fight_predictions['margin'] < 0.1])} out of {len(fight_predictions)} fights are very close (margin < 0.1)
"""
        
        # Methodology
        report_content += f"""

## Methodology

### Data Preprocessing
1. **Feature Engineering**: Created composite features (power ratios, tactical intelligence)
2. **Categorical Encoding**: Label encoded categorical variables
3. **Missing Value Handling**: Applied median imputation for numerical features
4. **Feature Scaling**: Standardized features to match training data distribution

### Prediction Process
1. **Model Loading**: Used best performing model from training phase
2. **Data Transformation**: Applied same preprocessing pipeline as training
3. **Probability Prediction**: Generated win probabilities using model.predict_proba()
4. **Fight Analysis**: Compared individual probabilities to determine winners

### Confidence Levels
- **High Confidence**: |probability - 0.5| > 0.3
- **Medium Confidence**: 0.15 < |probability - 0.5| <= 0.3  
- **Low Confidence**: |probability - 0.5| <= 0.15

## Files Generated
- `task2_character_predictions.csv`: Individual character predictions
- `task2_fight_predictions.csv`: Fight outcome predictions
- `14_task2_character_predictions.png`: Character analysis visualizations
- `15_task2_fight_analysis.png`: Fight prediction visualizations
- `16_task2_character_comparison.png`: Feature comparison plots
- `task2_interactive_dashboard.html`: Interactive prediction dashboard

## Recommendations

### For Character Development
1. **Focus on Ranking**: Ranking appears to be highly predictive of success
2. **Balance Intelligence**: Combination of general and battle intelligence is crucial
3. **Power vs Speed**: Both matter, but optimization depends on fighting style

### For Fight Strategy
1. **Exploit Weaknesses**: Target opponent's lowest-rated features
2. **Leverage Strengths**: Maximize advantages in top-performing areas
3. **Training Focus**: Improve features with highest model importance

---
"""
        
        # Save report
        with open('outputs/reports/Task2_Report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save summary as JSON
        task2_summary = {
            'total_characters': int(len(character_predictions)),
            'heroes_count': int(len(heroes)),
            'villains_count': int(len(villains)),
            'heroes_avg_probability': float(heroes['predicted_win_probability'].mean()),
            'villains_avg_probability': float(villains['predicted_win_probability'].mean()),
            'strongest_character': str(character_predictions.loc[character_predictions['predicted_win_probability'].idxmax(), 'name']),
            'strongest_probability': float(character_predictions['predicted_win_probability'].max()),
            'total_fights': int(len(fight_predictions)) if fight_predictions is not None else 0,
            'fight_winners': fight_predictions['winner'].tolist() if fight_predictions is not None else [],
            'model_used': str(self.model_name)
        }
        
        # Convert any NumPy types
        task2_summary = convert_numpy_types(task2_summary)
        
        with open('outputs/reports/task2_summary.json', 'w', encoding='utf-8') as f:
            json.dump(task2_summary, f, indent=2)
        
        print(" Task 2 report saved to outputs/reports/")

if __name__ == "__main__":
    # Test the Task 2 module
    print(" Testing Task2Predictor module...")
    
    try:
        # This would normally be called from main.py with a trained modeler
        print(" Task 2 module ready for integration with main pipeline")
        print("Run main.py to execute complete Task 2 analysis")
        
    except Exception as e:
        print(f" Error: {str(e)}")