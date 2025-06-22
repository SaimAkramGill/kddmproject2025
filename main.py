#!/usr/bin/env python3
"""
Superhero Winning Probability Prediction
Group 125 - Machine Learning Project

Usage: python main.py
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Import Classes
from EDA import SuperheroEDA
from preprocessing import DataPreprocessor
from modeling import SuperheroModeling
from task2 import Task2Predictor
from task3 import Task3Analyzer
from utils import ProjectUtils

def create_output_structure():
    """ Output directories """
    directories = [
        'outputs',
        'outputs/plots',
        'outputs/reports',
        'outputs/models',
        'outputs/predictions',
        'outputs/presentation'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Output directory structure created!")

def print_banner():
    """Print project banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                 SUPERHERO PREDICTION PROJECT                  ║
    ║                     Group 125 - 2025                          ║
    ║          Winning Probability Prediction & Analysis           ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def run_eda(data_path='data.csv', save_outputs=True):
    """Exploratory Data Analysis"""
    print("\n TASK: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize EDA class
    eda = SuperheroEDA(data_path)
    
    # Load and analyze data
    eda.load_data()
    
    # Run comprehensive EDA
    basic_stats = eda.basic_analysis()
    correlation_analysis = eda.correlation_analysis()
    feature_analysis = eda.feature_analysis()
    data_quality = eda.data_quality_analysis()
    
    if save_outputs:
        # Generate visualizations
        eda.create_overview_plots()
        eda.create_correlation_plots()
        eda.create_distribution_plots()
        eda.create_target_analysis_plots()
        
        # Generate EDA report
        eda.generate_eda_report()
    
    elapsed_time = time.time() - start_time
    print(f" EDA completed in {elapsed_time:.2f} seconds")
    
    return eda

def run_preprocessing(eda, save_outputs=True):
    """Data Preprocessing"""
    print("\n TASK: DATA PREPROCESSING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(eda.data)
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    if save_outputs:
        # Save preprocessing report
        preprocessor.generate_preprocessing_report()
        
        # Save processed data
        preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
    
    elapsed_time = time.time() - start_time
    print(f" Preprocessing completed in {elapsed_time:.2f} seconds")
    
    return preprocessor, X_train, X_test, y_train, y_test

def run_modeling(X_train, X_test, y_train, y_test, save_outputs=True):
    """Machine Learning Modeling"""
    print("\n TASK: MACHINE LEARNING MODELING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize modeling class
    modeler = SuperheroModeling()
    
    # Train and evaluate models
    results = modeler.train_models(X_train, X_test, y_train, y_test)
    
    # Model evaluation and selection
    best_model, model_metrics = modeler.evaluate_models(results)
    
    if save_outputs:
        # Generate model visualizations
        modeler.create_evaluation_plots(results)
        modeler.create_feature_importance_plots()
        modeler.create_performance_comparison_plots(results)
        
        # Generate modeling report
        modeler.generate_modeling_report(results, model_metrics)
        
        # Save best model
        modeler.save_model(best_model)
    
    elapsed_time = time.time() - start_time
    print(f" Modeling completed in {elapsed_time:.2f} seconds")
    
    return modeler

def run_task2(modeler, save_outputs=True):
    """Task 2: Character Predictions Starting"""
    print("\n TASK 2: CHARACTER WINNING PROBABILITY PREDICTIONS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize Task 2 predictor
    task2 = Task2Predictor(modeler)
    
    # Load Task 2 data
    task2.load_task2_data()
    
    # Make predictions
    character_predictions = task2.predict_characters()
    fight_predictions = task2.predict_fights()
    
    if save_outputs:
        # Generate Task 2 visualizations
        task2.create_prediction_plots()
        task2.create_fight_analysis_plots()
        task2.create_character_comparison_plots()
        
        # Generate Task 2 report
        task2.generate_task2_report(character_predictions, fight_predictions)
        
        # Save predictions
        task2.save_predictions(character_predictions, fight_predictions)
    
    elapsed_time = time.time() - start_time
    print(f" Task 2 completed in {elapsed_time:.2f} seconds")
    
    return task2

def run_task3(modeler, eda_data, save_outputs=True):
    """Task 3: Perfect Villain Analysis"""
    print("\n TASK 3: PERFECT VILLAIN ANALYSIS")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize Task 3 analyzer
    task3 = Task3Analyzer(modeler, eda_data)
    
    # Load Task 3 data
    task3.load_task3_data()
    
    # Analyze perfect villain
    statistical_analysis = task3.statistical_analysis()
    feature_analysis = task3.feature_dominance_analysis()
    comparative_analysis = task3.comparative_analysis()
    
    if save_outputs:
        # Generate Task 3 visualizations
        task3.create_statistical_plots()
        task3.create_comparison_plots()
        task3.create_radar_chart()
        task3.create_percentile_analysis()
        
        # Generate Task 3 report
        task3.generate_task3_report(statistical_analysis, feature_analysis)
        
        # Save analysis results
        task3.save_analysis_results()
    
    elapsed_time = time.time() - start_time
    print(f" Task 3 completed in {elapsed_time:.2f} seconds")
    
    return task3

def generate_final_presentation(eda, modeler, task2, task3):
    """Generate final presentation materials"""
    print("\n GENERATING FINAL PRESENTATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize utils
    utils = ProjectUtils()
    
    # Collect all results
    results = {
        'eda': eda,
        'modeler': modeler,
        'task2': task2,
        'task3': task3
    }
    
    # Generate presentation slides
    utils.generate_presentation_slides(results)
    
    # Generate executive summary
    utils.generate_executive_summary(results)
    
    # Generate final project report
    utils.generate_final_report(results)
    
    # Create presentation-ready plots
    utils.create_presentation_plots(results)
    
    elapsed_time = time.time() - start_time
    print(f" Presentation materials generated in {elapsed_time:.2f} seconds")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Superhero Prediction Project')
    parser.add_argument('--data-path', default='data.csv', help='Path to main dataset')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA task')
    parser.add_argument('--skip-task2', action='store_true', help='Skip Task 2')
    parser.add_argument('--skip-task3', action='store_true', help='Skip Task 3')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save outputs')
    parser.add_argument('--quick', action='store_true', help='Quick run with minimal outputs')
    
    args = parser.parse_args()
    
    # Setup
    print_banner()
    create_output_structure()
    
    total_start_time = time.time()
    save_outputs = not args.no_save
    
    # Initialize results storage
    eda = None
    modeler = None
    task2 = None
    task3 = None
    
    try:
        # Task 1: EDA
        if not args.skip_eda:
            eda = run_eda(args.data_path, save_outputs)
        else:
            print("Skipping EDA...")
            # Load minimal data for other tasks
            eda = SuperheroEDA(args.data_path)
            eda.load_data()
        
        # Task: Preprocessing & Modeling
        preprocessor, X_train, X_test, y_train, y_test = run_preprocessing(eda, save_outputs)
        modeler = run_modeling(X_train, X_test, y_train, y_test, save_outputs)
        
        # Task 2: Character Predictions
        if not args.skip_task2:
            task2 = run_task2(modeler, save_outputs)
        else:
            print("Skipping Task 2...")
        
        # Task 3: Perfect Villain Analysis
        if not args.skip_task3:
            task3 = run_task3(modeler, eda.data, save_outputs)
        else:
            print(" Skipping Task 3...")
        
        # Generate final presentation materials
        if save_outputs and not args.quick:
            generate_final_presentation(eda, modeler, task2, task3)
        
        # Final summary
        total_elapsed = time.time() - total_start_time
        
        print(f"\n PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f" Total execution time: {total_elapsed:.2f} seconds")
        print(f" All outputs saved to: ./outputs/")
        print(f" Presentation materials: ./outputs/presentation/")
        print(f" Model performance: {modeler.best_auc:.4f} AUC" if modeler else "")
        print(f" Task 2 predictions: {' Complete' if task2 else ' Skipped'}")
        print(f" Task 3 analysis: {'Complete' if task3 else 'Skipped'}")
        
        # Quick results summary
        if modeler:
            print(f"\n QUICK RESULTS SUMMARY:")
            print(f"• Best Model: {modeler.best_model_name}")
            print(f"• Test AUC: {modeler.best_auc:.4f}")
            print(f"• Dataset: {len(eda.data)} characters")
            print(f"• Top Feature: {modeler.top_features[0] if hasattr(modeler, 'top_features') else 'N/A'}")
        

        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("Check data files and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()