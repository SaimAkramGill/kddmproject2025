# Superhero Winning Probability Prediction
## Group 125 Presentation
### June 22, 2025

---

## Slide 1: Exploratory Data Analysis - Overview

### Dataset Characteristics
- **Dataset Size**: 1,041 characters, 24 features
- **Target Variable**: Winning probability (0.22 to 15.20) → Binary classification (win/lose)
- **Feature Types**: 12 numerical, 11 categorical features
- **Class Balance**: 73.6% winners, 26.4% losers (imbalanced)

### Key Findings from EDA
- **Missing Data**: High in eye_color (32.4%), secret_code (37.3%), speed (27.2%)
- **Data Quality Issues**: Role spellings inconsistent (Hero vs H3ro vs HerO)
- **Outliers**: Some extreme win_prob values (max: 15.20 vs typical 0.2-1.0)

### Target Distribution
- Winning probability ranges from 0.22 to 15.20 (capped at 1.0 for modeling)
- 73.6% of characters have winning probability > 0.5
- Clear class imbalance favoring winners

---

## Slide 2: Exploratory Data Analysis - Feature Insights

### Most Important Features (Correlation with Win Probability)
1. **ranking**: r = -0.0690 (lower ranking = higher win probability)
2. **battle_iq**: r = +0.0470 (higher battle IQ = higher win probability)
3. **intelligence**: r = +0.0430 (smarter characters tend to win)
4. **speed**: r = +0.0425 (faster characters have slight advantage)
5. **height**: r = -0.0279 (shorter characters tend to win slightly more)


### Notable Patterns Discovered
- **Character Roles**: Heroes (288) vs Villains (292) roughly balanced, but multiple spellings
- **Power vs Success**: Surprisingly weak correlation between raw power and winning
- **Intelligence Matters**: Battle IQ and intelligence are key predictors
- **Ranking is King**: Most important single predictor (popularity/effectiveness measure)

---

## Slide 3: Methodology

### Data Preprocessing Pipeline
- **Missing Value Imputation**: Median imputation for numerical features, mode for categorical
- **Categorical Encoding**: Label encoding for categorical variables
- **Role Standardization**: Unified Hero/Villain spellings (H3ro → Hero, VIllain → Villain)
- **Target Engineering**: Capped extreme values, binary classification (threshold = 0.5)

### Data Splitting Strategy
- **Training Set**: 80% of data (828 samples)
- **Test Set**: 20% of data (208 samples)  
- **Stratified Split**: Maintained 73.9% winner / 26.1% loser balance
- **Cross-Validation**: 5-fold stratified CV for model selection

### Models Evaluated
- **Random Forest**: Tree-based ensemble with class balancing
- **Gradient Boosting**: Sequential boosting for complex patterns  
- **Logistic Regression**: Linear baseline with feature scaling
- **SVM**: Support Vector Machine with RBF kernel
- **AdaBoost**: Adaptive boosting ensemble
- **K-Nearest Neighbors**: Distance-based classification

---

## Slide 4: Model Evaluation

### Model Performance Comparison
| Model | Cross-Val AUC | Test AUC | Best Model |
|-------|---------------|----------|------------|
| AdaBoost | 0.960 | 0.954 | ✓ |
| Gradient Boosting | 0.916 | 0.919 | ✗ |
| Random Forest | 0.892 | 0.892 | ✗ |
| SVM | 0.838 | 0.834 | ✗ |
| Logistic Regression | 0.812 | 0.802 | ✗ |
| K-Nearest Neighbors | 0.732 | 0.687 | ✗ |


### Best Model: AdaBoost
- **Final Test AUC**: 0.954 - Excellent performance given weak feature correlations
- **Precision**: 0.914 - 91% of predicted winners actually win  
- **Recall**: 0.961 - Captures 96% of actual winners
- **F1-Score**: 0.937 - Balanced performance metric

### Feature Importance (Top 5)
1. **skin_type_encoded**: 0.350 - Important predictor
2. **special_attack_encoded**: 0.191 - Important predictor
3. **training_efficiency**: 0.073 - Important predictor
4. **power_level**: 0.060 - Raw power still important
5. **gender**: 0.060 - Important predictor

---

## Slide 5: Discussion & Task Results

### Task 2: Unseen Data Predictions
**Three Superheroes:**
- 🦸 Captain Britain: 0.657 (65.7%)
- 🦸 Golden Glider: 0.743 (74.3%)

**Three Villains:**
- 🦹 Overhaul: 0.755 (75.5%)
- 🦹 Madame Hydra: 0.741 (74.1%)
- 🦹 King Shark: 0.652 (65.2%)

**Fight Predictions:**
- Fight 1: Endeavor vs Overhaul → **Overhaul WINS** (0.755 vs 0.547)
- Fight 2: Captain Britain vs Madame Hydra → **Madame Hydra WINS** (0.741 vs 0.657)
- Fight 3: Golden Glider vs King Shark → **Golden Glider WINS** (0.743 vs 0.652)

### Task 3: Perfect Villain Analysis (100% Win Rate)
**Why Penguin Always Wins:**
- **Power Level**: 16,015 (91st percentile) - Near maximum power
- **Excellent Ranking**: 1,218 (23rd percentile) - Top-tier popularity/effectiveness  
- **Extensive Training**: 9,496 hours (92nd percentile) - Exceptional preparation
- **Balanced Stats**: Above average in speed, intelligence, and battle IQ

**Key Insight**: Penguin doesn't dominate any single category but achieves an optimal combination of high power, excellent ranking, and extensive training - creating an unbeatable profile.

### Special Observations
- **Surprising Finding**: Raw power alone doesn't guarantee victory - ranking and intelligence matter more
- **Model Limitations**: Weak correlations suggest other unmeasured factors influence winning
- **Data Quality**: Inconsistent spellings and missing values indicate synthetic data artifacts
- **Real-World Applicability**: Framework could apply to sports analytics, gaming balance, or competitive rankings

---

## Thank You!
### Questions & Discussion

**Contact Information:**
- Project Repository: [\[GitHub Link\]](https://github.com/SaimAkramGill/kddmproject2025.git)
- Email: [muhammad.akram@edu.uni-graz.at]

**Key Takeaways:**
1. Ranking and intelligence are more predictive than raw power
2. AdaBoost achieved best performance
3. Perfect villains win through balanced excellence, not single strengths
4. Data quality and feature engineering are crucial for model success
