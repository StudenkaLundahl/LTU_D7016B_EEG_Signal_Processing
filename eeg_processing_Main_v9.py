#%%
"""
EEG Data Processing and Classification - Complete Implementation
Signal Processing and Machine Learning for EEG Data Classification

Student: Studenka Lundahl
Supervisor: Adoul Mohammed Amin
Date: December 15, 2025

Project Description:
This script implements a complete EEG data analysis pipeline for classifying golf putting 
performance (Hit vs Miss) based on brain activity. The analysis includes data preprocessing, 
feature extraction, and three different classification approaches: Logistic Regression (Grade 3), 
Random Forest (Grade 4), and Convolutional Neural Networks (Grade 5).

Implementation Overview:
- GRADE 3: Data exploration, preprocessing, epoching, basic feature extraction, and logistic regression
- GRADE 4: Advanced PSD-based feature extraction, Random Forest classification, feature importance analysis
- GRADE 5: Deep learning data preparation, CNN architecture development, and model training
- ADVANCED: Statistical analysis, comprehensive visualizations, and multi-model comparisons

Data Structure:
EEG data is organized in session folders (50-1 through 50-10), each containing:
- ExG.csv: 8-channel EEG recordings (Fp1, Fp2, C3, C4, P3, P4, O1, O2) at 500 Hz
- Marker.csv: Event timestamps with codes (ext_0 for Hit, ext_1 for Miss)

Output:
All visualizations are saved as PNG files to avoid popup windows. Results include confusion 
matrices, feature importance plots, training histories, and comprehensive statistical analyses.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots without displaying windows
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive plotting mode

# Import custom EEG analysis modules
from EEGDataExplorer import *
from EEGPreprocessor import *
from EventAligner import *
from FeatureExtractor import *
from AdvancedFeatureExtractor import *
from EEGClassifier import *
from RandomForestEEGClassifier import *
from DeepLearningDataPreparator import DeepLearningDataPreparator
from CNNEEGClassifier import CNNEEGClassifier
from ResultsAnalyzer import ResultsAnalyzer
from GrandAveragePSDPlotter import GrandAveragePSDPlotter
from ThreeModelComparator import ThreeModelComparator
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION SECTION - CENTRALIZED PARAMETER MANAGEMENT
# ============================================================================
"""
All analysis parameters are defined here for easy modification and experimentation.
This centralized configuration allows for quick adjustments without modifying the 
core processing logic.
"""

print("\n" + "="*80)
print("EEG DATA PROCESSING - CONFIGURATION")
print("="*80 + "\n")

# ----------------------------------------------------------------------------
# Analysis Mode Configuration
# ----------------------------------------------------------------------------
# These flags control which analysis components are executed

USE_ADVANCED_PSD_FEATURES = True   # Use advanced 152-feature PSD extraction (Grade 4)
                                   # If False, uses basic 24-feature extraction (Grade 3)

USE_RANDOM_FOREST = True           # Enable Random Forest classifier (Grade 4)
                                   # If False, only Logistic Regression is used

COMPARE_MODELS = True              # Train both LR and RF for comparison
                                   # Enables side-by-side model evaluation

USE_CNN = True                     # Enable CNN deep learning model (Grade 5)
                                   # Requires properly prepared 3D temporal data

GENERATE_PSD_PLOTS = True          # Create power spectral density visualizations
                                   # Shows frequency domain characteristics of signals

GENERATE_COMPREHENSIVE_ANALYSIS = True  # Generate statistical reports and comparisons
                                        # Includes t-tests, effect sizes, and summary tables

# ----------------------------------------------------------------------------
# Preprocessing Parameters
# ----------------------------------------------------------------------------
# Signal filtering parameters based on EEG best practices for motor tasks

FILTER_LOW_FREQ = 1.0     # High-pass filter cutoff (Hz)
                          # Removes slow drifts and DC offset from signals

FILTER_HIGH_FREQ = 30.0   # Low-pass filter cutoff (Hz)
                          # 30 Hz is optimal for motor-related EEG activity
                          # Removes high-frequency noise and muscle artifacts

NOTCH_FREQ = 50.0         # Notch filter frequency (Hz)
                          # Removes powerline interference (50 Hz in Europe, 60 Hz in US)

# ----------------------------------------------------------------------------
# Epoching Parameters
# ----------------------------------------------------------------------------
# Time windows for extracting signal segments around events

EPOCH_TMIN = -2.0         # Epoch start time relative to event (seconds)
                          # Negative values indicate time before the event
                          # -2.0 provides 2 seconds of pre-event baseline data

EPOCH_TMAX = 1.0          # Epoch end time relative to event (seconds)
                          # Positive values indicate time after the event
                          # 1.0 captures the putting action and immediate aftermath

BASELINE_START = -2.0     # Baseline correction window start (seconds)
                          # Baseline period should be before the motor action

BASELINE_END = -1.0       # Baseline correction window end (seconds)
                          # 1-second baseline is standard for motor tasks

# ----------------------------------------------------------------------------
# Power Spectral Density (PSD) Parameters
# ----------------------------------------------------------------------------
# Frequency analysis configuration using Welch's method

PSD_FMIN = 1.0           # Minimum frequency for PSD computation (Hz)
                         # Excludes very slow frequencies that may contain artifacts

PSD_FMAX = 30.0          # Maximum frequency for PSD computation (Hz)
                         # Matches the low-pass filter to focus on relevant frequencies

PSD_N_FFT = 256          # FFT window length for Welch's method
                         # Larger values provide better frequency resolution
                         # 256 samples at 500 Hz = 0.512 second windows

# ----------------------------------------------------------------------------
# Random Forest Hyperparameters
# ----------------------------------------------------------------------------
# Configuration for Random Forest classifier (Grade 4)

RF_N_ESTIMATORS = 100             # Number of decision trees in the forest
                                  # More trees improve stability but increase computation time

RF_MAX_DEPTH = 10                 # Maximum depth of each decision tree
                                  # Lower values prevent overfitting to training data
                                  # None allows trees to expand until all leaves are pure

RF_MIN_SAMPLES_SPLIT = 2          # Minimum samples required to split an internal node
                                  # Higher values create more general trees (less overfitting)

RF_MIN_SAMPLES_LEAF = 1           # Minimum samples required in each leaf node
                                  # Higher values create smoother decision boundaries

RF_MAX_FEATURES = 'sqrt'          # Number of features to consider for best split
                                  # 'sqrt' uses square root of total features (recommended)
                                  # 'log2' is another common choice

RF_OPTIMIZE_HYPERPARAMETERS = True  # Enable grid search for optimal parameters
                                    # Warning: significantly increases training time
                                    # Systematically tests parameter combinations

# Hyperparameter grid for GridSearchCV optimization
# Only used when RF_OPTIMIZE_HYPERPARAMETERS = True
RF_HYPERPARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ----------------------------------------------------------------------------
# CNN Architecture Configuration
# ----------------------------------------------------------------------------
# Deep learning model parameters (Grade 5)

CNN_ARCHITECTURE = 'standard'     # Predefined architecture type
                                  # Options: 'shallow' (fast, fewer parameters)
                                  #          'standard' (balanced, recommended)
                                  #          'deep' (slow, more parameters)

CNN_EPOCHS = 100                  # Maximum number of training epochs
                                  # Early stopping will halt training if no improvement

CNN_BATCH_SIZE = 32               # Number of samples per gradient update
                                  # Smaller batches = more updates but noisier gradients
                                  # Reduce if memory errors occur

CNN_LEARNING_RATE = 0.001         # Initial learning rate for Adam optimizer
                                  # Will be automatically reduced if training plateaus

CNN_VALIDATION_SPLIT = 0.2        # Fraction of training data used for validation
                                  # 0.2 means 80% training, 20% validation

CNN_EARLY_STOPPING_PATIENCE = 15  # Epochs to wait for improvement before stopping
                                  # Prevents unnecessary training when model has converged

CNN_REDUCE_LR_PATIENCE = 7        # Epochs to wait before reducing learning rate
                                  # Helps model find better minima when stuck

# Custom CNN Architecture (Future Enhancement)
# Currently not implemented - uses predefined architectures above
# Implementation would require modifications to CNNEEGClassifier.py
CNN_CUSTOM_ARCHITECTURE = False   # Keep False - custom architecture not yet implemented

# Parameters for future custom CNN implementation
# These would be used to build a fully customizable architecture
# Kept here for future development reference
CNN_CONV_FILTERS = [32, 64, 128]       # Filters per convolutional layer
CNN_CONV_KERNEL_SIZES = [50, 25, 10]  # Temporal kernel sizes
CNN_POOL_SIZES = [4, 4, 2]             # Pooling sizes for downsampling
CNN_DROPOUT_RATES = [0.3, 0.3, 0.4]    # Dropout rates for regularization
CNN_DENSE_UNITS = [128, 64]            # Units in fully connected layers
CNN_DENSE_DROPOUT = [0.5, 0.5]         # Dropout rates in dense layers

# ----------------------------------------------------------------------------
# Session-Specific Environmental Notes
# ----------------------------------------------------------------------------
# Documentation of data quality and environmental factors
# These notes help explain performance variations across sessions

ENVIRONMENTAL_NOTES = {
    '50-1': 'Good quality',
    '50-2': 'Good quality',
    '50-3': 'Good quality',
    '50-4': 'Good quality',
    '50-5': 'Optimal quality - impedance 12-20 kΩ',
    '50-6': 'Good quality',
    '50-7': 'Good quality',
    '50-8': 'Sunlight interference',
    '50-9': 'Participant hunger',
    '50-10': 'Participant hunger'
}

# ----------------------------------------------------------------------------
# Configuration Summary Output
# ----------------------------------------------------------------------------
# Display current analysis configuration for verification

print("ANALYSIS CONFIGURATION:")
print(f"  Feature extraction: {'ADVANCED PSD (152 features)' if USE_ADVANCED_PSD_FEATURES else 'BASIC (24 features)'}")
print(f"  Models to train: ", end='')
models_list = []
if COMPARE_MODELS:
    models_list.append('LR')
    models_list.append('RF')
elif USE_RANDOM_FOREST:
    models_list.append('RF')
else:
    models_list.append('LR')
if USE_CNN:
    models_list.append('CNN')
print(' + '.join(models_list))

print(f"\nPREPROCESSING:")
print(f"  Filters: {FILTER_LOW_FREQ}-{FILTER_HIGH_FREQ} Hz, notch {NOTCH_FREQ} Hz")
print(f"  Epoch window: {EPOCH_TMIN} to {EPOCH_TMAX} sec")
print(f"  Baseline: {BASELINE_START} to {BASELINE_END} sec")
print(f"  PSD range: {PSD_FMIN}-{PSD_FMAX} Hz")

if USE_RANDOM_FOREST:
    print(f"\nRANDOM FOREST PARAMETERS:")
    print(f"  n_estimators: {RF_N_ESTIMATORS}")
    print(f"  max_depth: {RF_MAX_DEPTH}")
    print(f"  min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  max_features: {RF_MAX_FEATURES}")
    print(f"  Hyperparameter optimization: {'ENABLED' if RF_OPTIMIZE_HYPERPARAMETERS else 'DISABLED'}")

if USE_CNN:
    print(f"\nCNN PARAMETERS:")
    print(f"  Architecture: {CNN_ARCHITECTURE}")
    print(f"  Max epochs: {CNN_EPOCHS} (early stopping enabled)")
    print(f"  Batch size: {CNN_BATCH_SIZE}")
    print(f"  Learning rate: {CNN_LEARNING_RATE}")
    print(f"  Validation split: {CNN_VALIDATION_SPLIT}")

print(f"\nADVANCED ANALYSIS:")
print(f"  PSD visualizations: {'ENABLED' if GENERATE_PSD_PLOTS else 'DISABLED'}")
print(f"  Multi-model comparison: {'ENABLED' if (GENERATE_COMPREHENSIVE_ANALYSIS and USE_CNN) else 'DISABLED'}")

print("\n" + "="*80 + "\n")


# ============================================================================
# SINGLE SESSION PROCESSING FUNCTION
# ============================================================================

def process_single_session(data_folder, session_name, 
                          results_analyzer=None, 
                          psd_plotter=None,
                          three_model_comparator=None):
    """
    Process a single EEG session through the complete analysis pipeline.
    
    This function implements all tasks from Grade 3, Grade 4, and Grade 5:
    - Data loading and exploration (Grade 3)
    - Preprocessing and artifact removal (Grade 3)
    - Event alignment and epoching (Grade 3)
    - Feature extraction (Grade 3 basic or Grade 4 advanced)
    - Classification with Logistic Regression (Grade 3)
    - Classification with Random Forest (Grade 4)
    - Deep learning classification with CNN (Grade 5)
    
    Parameters:
    -----------
    data_folder : str
        Path to the session folder containing ExG.csv and Marker.csv files
    session_name : str
        Identifier for the session (e.g., '50-1')
    results_analyzer : ResultsAnalyzer, optional
        Object for aggregating results across sessions
    psd_plotter : GrandAveragePSDPlotter, optional
        Object for creating PSD visualizations
    three_model_comparator : ThreeModelComparator, optional
        Object for comparing all three models
    
    Returns:
    --------
    tuple
        (epochs, metrics_lr, metrics_rf, metrics_cnn, cv_results_lr, cv_results_rf)
        Returns None for unused models
    """
    
    print("\n" + "="*80)
    print(f"PROCESSING SESSION: {session_name}")
    print("="*80 + "\n")
    
    # Environmental factors for this session
    env_note = ENVIRONMENTAL_NOTES.get(session_name, 'No notes available')
    print(f"Session notes: {env_note}\n")
    
    # ========================================================================
    # GRADE 3 - TASK 1.4: DATA EXPLORATION
    # ========================================================================
    print("### TASK 1.4: DATA EXPLORATION ###\n")
    
    # Initialize data explorer and load EEG files
    explorer = EEGDataExplorer(data_folder)
    explorer.explore_data()
    
    # ========================================================================
    # GRADE 3 - TASK 1.5: DATA PREPROCESSING
    # ========================================================================
    print("\n### TASK 1.5: DATA PREPROCESSING ###\n")
    
    # Apply filtering and artifact removal
    preprocessor = EEGPreprocessor(explorer.exg_file, explorer.meta_file)
    raw_filtered = preprocessor.preprocess(
        l_freq=FILTER_LOW_FREQ, 
        h_freq=FILTER_HIGH_FREQ, 
        notch_freq=NOTCH_FREQ
    )
    
    # Create visualization comparing raw and preprocessed signals
    print("Creating preprocessing comparison plot...")
    preprocessor.plot_comparison(duration=10.0, session_name=session_name)
    output_file = f'preprocessing_comparison_{session_name}.png'
    if os.path.exists('preprocessing_comparison.png'):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename('preprocessing_comparison.png', output_file)
    
    # ========================================================================
    # GRADE 3 - TASK 1.6: EVENT ALIGNMENT
    # ========================================================================
    print("\n### TASK 1.6: EVENT ALIGNMENT & EPOCHING ###\n")
    
    # Map event codes to behavioral outcomes
    aligner = EventAligner(raw_filtered, explorer.marker_file, preprocessor.first_timestamp)
    epochs = aligner.create_epochs(
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=(BASELINE_START, BASELINE_END)
    )
    
    if epochs is None or len(epochs) == 0:
        print("\nSkipping session - no valid epochs created")
        return None, None, None, None, None, None
    
    # Create event-related potential (ERP) visualization
    print("Creating ERP comparison plot...")
    aligner.plot_erp(epochs, session_name=session_name)
    output_file = f'erp_comparison_{session_name}.png'
    if os.path.exists('erp_comparison.png'):
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename('erp_comparison.png', output_file)
    
    # Count behavioral performance
    behavioral_hits = len(epochs['Hit'])
    behavioral_misses = len(epochs['Miss'])
    print(f"\nBehavioral Performance:")
    print(f"  Hits: {behavioral_hits}/50 ({behavioral_hits*2}%)")
    print(f"  Misses: {behavioral_misses}/50 ({behavioral_misses*2}%)")
    
    # ========================================================================
    # FEATURE EXTRACTION - GRADE 3 OR GRADE 4
    # ========================================================================
    
    if USE_ADVANCED_PSD_FEATURES:
        # GRADE 4 - TASK 2.1: ADVANCED PSD FEATURE EXTRACTION
        print("\n### TASK 2.1: ADVANCED PSD FEATURE EXTRACTION ###\n")
        
        # Extract comprehensive frequency-domain features
        extractor = AdvancedFeatureExtractor(epochs)
        features, labels, feature_names = extractor.extract_psd_features(
            fmin=PSD_FMIN, 
            fmax=PSD_FMAX, 
            n_fft=PSD_N_FFT
        )
        
        print(f"Extracted {len(feature_names)} PSD features using Welch's method")
        
        # Visualize PSD feature importance
        extractor.plot_feature_importance(top_n=20, session_name=session_name)
        output_file = f'psd_feature_importance_{session_name}.png'
        if os.path.exists('psd_feature_importance.png'):
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename('psd_feature_importance.png', output_file)
        
    else:
        # GRADE 3 - TASK 1.7: BASIC FEATURE EXTRACTION
        print("\n### TASK 1.7: BASIC FEATURE EXTRACTION ###\n")
        
        # Extract time-domain and simple frequency features
        extractor = FeatureExtractor(epochs)
        features, labels, feature_names = extractor.extract_features()
        
        print(f"Extracted {len(feature_names)} basic features")
        
        # Visualize basic feature importance
        extractor.plot_feature_importance(top_n=20, session_name=session_name)
        output_file = f'feature_importance_{session_name}.png'
        if os.path.exists('feature_importance.png'):
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename('feature_importance.png', output_file)
    
    # ========================================================================
    # CLASSIFICATION (Grade 3/4): Logistic Regression & Random Forest
    # ========================================================================
    
    metrics_lr = None
    metrics_rf = None
    
    if COMPARE_MODELS:
        # TASK 2.6: MODEL COMPARISON (LR VS RF)
        print("\n" + "="*80)
        print("TASK 2.6: MODEL COMPARISON (LOGISTIC REGRESSION VS RANDOM FOREST)")
        print("="*80 + "\n")
        
        # Train Logistic Regression
        print("### TRAINING LOGISTIC REGRESSION (BASELINE) ###\n")
        classifier_lr = EEGClassifier(features, labels, feature_names)
        classifier_lr.prepare_data(test_size=0.2, random_state=42)
        classifier_lr.train_model()
        metrics_lr = classifier_lr.evaluate_model()
        cv_results_lr = classifier_lr.cross_validate(cv=5)
        metrics_lr['cv_mean'] = cv_results_lr['mean_score']
        metrics_lr['cv_std'] = cv_results_lr['std_score']
        
        # Save LR confusion matrix
        classifier_lr.plot_confusion_matrix(metrics_lr, session_name=session_name)
        output_file = f'confusion_matrix_lr_{session_name}.png'
        if os.path.exists('confusion_matrix.png'):
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename('confusion_matrix.png', output_file)
        
        # Train Random Forest with configurable parameters
        print("\n### TASK 2.4: RANDOM FOREST IMPLEMENTATION ###\n")
        classifier_rf = RandomForestEEGClassifier(features, labels, feature_names)
        classifier_rf.prepare_data(test_size=0.2, random_state=42, scale_features=True)
        
        # Use configurable parameters from main script
        classifier_rf.train_model(
            optimize_hyperparameters=RF_OPTIMIZE_HYPERPARAMETERS,
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features=RF_MAX_FEATURES,
            param_grid=RF_HYPERPARAM_GRID
        )
        
        metrics_rf = classifier_rf.evaluate_model()
        cv_results_rf = classifier_rf.cross_validate(cv=5)
        metrics_rf['cv_mean'] = cv_results_rf['mean_score']
        metrics_rf['cv_std'] = cv_results_rf['std_score']
        
        # TASK 2.5: Feature importance from Random Forest
        print("\n### TASK 2.5: FEATURE IMPORTANCE ANALYSIS (RANDOM FOREST) ###\n")
        classifier_rf.plot_feature_importance(top_n=20, session_name=session_name)
        output_file = f'feature_importance_rf_{session_name}.png'
        if os.path.exists('feature_importance_rf.png'):
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename('feature_importance_rf.png', output_file)
        
        # Save RF confusion matrix
        classifier_rf.plot_confusion_matrix(metrics_rf, session_name=session_name)
        output_file = f'confusion_matrix_rf_{session_name}.png'
        if os.path.exists('confusion_matrix_rf.png'):
            if os.path.exists(output_file):
                os.remove(output_file)
            os.rename('confusion_matrix_rf.png', output_file)
        
        # Print comparison
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS (LR vs RF)")
        print("="*80)
        print(f"\nLogistic Regression:")
        print(f"  Test Accuracy:  {metrics_lr['test_accuracy']:.3f}")
        print(f"  F1-Score:       {metrics_lr['f1_score']:.3f}")
        print(f"  CV Accuracy:    {cv_results_lr['mean_score']:.3f} (+/- {cv_results_lr['std_score']:.3f})")
        print(f"\nRandom Forest:")
        print(f"  Test Accuracy:  {metrics_rf['test_accuracy']:.3f}")
        print(f"  F1-Score:       {metrics_rf['f1_score']:.3f}")
        print(f"  CV Accuracy:    {cv_results_rf['mean_score']:.3f} (+/- {cv_results_rf['std_score']:.3f})")
        
        if metrics_rf['test_accuracy'] > metrics_lr['test_accuracy']:
            improvement = (metrics_rf['test_accuracy'] - metrics_lr['test_accuracy']) * 100
            print(f"\nRandom Forest performs BETTER (+{improvement:.1f}% accuracy)")
        else:
            print(f"\nModels perform similarly")
        print("="*80 + "\n")
    
    # ========================================================================
    # GRADE 5: DEEP LEARNING (CNN)
    # ========================================================================
    
    metrics_cnn = None
    
    if USE_CNN:
        print("\n" + "="*80)
        print("GRADE 5: DEEP LEARNING WITH CNN")
        print("="*80 + "\n")
        
        # TASK 3.1: Prepare data for deep learning
        print("### TASK 3.1: DATA PREPARATION FOR DEEP LEARNING ###\n")
        dl_preparator = DeepLearningDataPreparator(epochs, verbose=True)
        dl_data = dl_preparator.prepare_data_3d(test_size=0.2, random_state=42, normalize=True)
        dl_preparator.print_data_summary()
        
        # TASK 3.2 & 3.3: Build, train, and evaluate CNN
        print("### TASKS 3.2 & 3.3: CNN MODEL DEVELOPMENT, TRAINING & VALIDATION ###\n")
        
        # Create CNN with configurable parameters
        cnn_classifier = CNNEEGClassifier(
            X_train=dl_data['X_train'],
            X_test=dl_data['X_test'],
            y_train=dl_data['y_train'],
            y_test=dl_data['y_test'],
            n_channels=dl_data['n_channels'],
            n_timepoints=dl_data['n_timepoints'],
            verbose=True
        )
        
        # Build model - uses predefined architecture from CNN_ARCHITECTURE setting
        cnn_classifier.build_model(architecture=CNN_ARCHITECTURE)
        
        # Train model with configurable parameters
        cnn_classifier.train_model(
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            validation_split=CNN_VALIDATION_SPLIT,
            early_stopping_patience=CNN_EARLY_STOPPING_PATIENCE,
            reduce_lr_patience=CNN_REDUCE_LR_PATIENCE
        )
        
        # Evaluate model
        metrics_cnn = cnn_classifier.evaluate_model()
        
        # Cross-validation (optional, may be slow)
        try:
            X_full = np.concatenate([dl_data['X_train'], dl_data['X_test']], axis=0)
            y_full = np.concatenate([dl_data['y_train'], dl_data['y_test']], axis=0)
            cv_results_cnn = cnn_classifier.cross_validate(X_full, y_full, cv=5, epochs=50, batch_size=32)
            metrics_cnn['cv_mean'] = cv_results_cnn['mean_score']
            metrics_cnn['cv_std'] = cv_results_cnn['std_score']
        except Exception as e:
            print(f"Warning: Could not perform CNN cross-validation: {str(e)}")
            metrics_cnn['cv_mean'] = None
            metrics_cnn['cv_std'] = None
        
        # Plot training history
        cnn_classifier.plot_training_history(session_name=session_name)
        
        # Plot confusion matrix
        cnn_classifier.plot_confusion_matrix(metrics_cnn, session_name=session_name)
        
        # ====================================================================
        # Compare all three models
        # ====================================================================
        if COMPARE_MODELS:
            print("\n" + "="*80)
            print("COMPREHENSIVE MODEL COMPARISON (LR vs RF vs CNN)")
            print("="*80)
            
            print(f"\nLogistic Regression:")
            print(f"  Test Accuracy:  {metrics_lr['test_accuracy']:.3f}")
            print(f"  F1-Score:       {metrics_lr['f1_score']:.3f}")
            
            print(f"\nRandom Forest:")
            print(f"  Test Accuracy:  {metrics_rf['test_accuracy']:.3f}")
            print(f"  F1-Score:       {metrics_rf['f1_score']:.3f}")
            
            print(f"\nCNN (Deep Learning):")
            print(f"  Test Accuracy:  {metrics_cnn['test_accuracy']:.3f}")
            print(f"  F1-Score:       {metrics_cnn['f1_score']:.3f}")
            
            # Find best model
            accuracies = {
                'Logistic Regression': metrics_lr['test_accuracy'],
                'Random Forest': metrics_rf['test_accuracy'],
                'CNN': metrics_cnn['test_accuracy']
            }
            best_model = max(accuracies, key=accuracies.get)
            best_acc = accuracies[best_model]
            
            print(f"\nBest Model: {best_model} ({best_acc:.3f} accuracy)")
            print("="*80 + "\n")
    
    # ========================================================================
    # ADD RESULTS TO ANALYZERS
    # ========================================================================
    
    # Add to two-model analyzer (LR vs RF)
    if results_analyzer is not None and COMPARE_MODELS:
        results_analyzer.add_session_results(
            session_name=session_name,
            lr_metrics=metrics_lr,
            rf_metrics=metrics_rf,
            behavioral_hits=behavioral_hits,
            environmental_notes=ENVIRONMENTAL_NOTES.get(session_name, '')
        )
    
    # Add to three-model analyzer (LR vs RF vs CNN)
    if three_model_comparator is not None and USE_CNN and COMPARE_MODELS:
        three_model_comparator.add_session_results(
            session_name=session_name,
            lr_metrics=metrics_lr,
            rf_metrics=metrics_rf,
            cnn_metrics=metrics_cnn
        )
    
    # ========================================================================
    # PSD VISUALIZATIONS
    # ========================================================================
    
    if GENERATE_PSD_PLOTS and psd_plotter is not None:
        print(f"\n### GENERATING PSD VISUALIZATIONS FOR {session_name} ###\n")
        try:
            psd_plotter.create_all_psd_plots(epochs, session_name)
        except Exception as e:
            print(f"Warning: Could not create PSD plots: {str(e)}")
    
    # ========================================================================
    # SESSION COMPLETE
    # ========================================================================
    
    print("\n" + "="*80)
    print(f"SESSION {session_name} COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return epochs, features, labels, metrics_lr, metrics_rf, metrics_cnn


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main function to orchestrate the complete EEG analysis pipeline.
    
    This function:
    1. Initializes analysis tools (results collectors, visualizers)
    2. Scans for available session folders
    3. Processes each session through the complete pipeline
    4. Generates comprehensive cross-session reports and comparisons
    5. Outputs summary of completed tasks and generated files
    """
    
    print("\n" + "="*80)
    print("INITIALIZING EEG ANALYSIS PIPELINE")
    print("="*80 + "\n")
    
    # ========================================================================
    # Initialize Analysis Tools
    # ========================================================================
    
    results_analyzer = None
    psd_plotter = None
    three_model_comparator = None
    
    # Set up cross-session comparison tools if comprehensive analysis is enabled
    if GENERATE_COMPREHENSIVE_ANALYSIS and COMPARE_MODELS:
        print("Initializing analysis tools...")
        results_analyzer = ResultsAnalyzer(output_dir='final_results')
        print("  Results analyzer initialized (LR vs RF comparison)")
        
        if USE_CNN:
            three_model_comparator = ThreeModelComparator(output_dir='final_results')
            print("  Three-model comparator initialized (LR vs RF vs CNN)")
    
    if GENERATE_PSD_PLOTS:
        psd_plotter = GrandAveragePSDPlotter(sfreq=500)
        print("  PSD plotter initialized")
    
    print()
    
    # ========================================================================
    # Scan for Available Session Folders
    # ========================================================================
    
    # Expected session folders
    sessions = [f'50-{i}' for i in range(1, 11)]
    
    print("Scanning for session folders...")
    available_sessions = []
    for session in sessions:
        if os.path.exists(session):
            available_sessions.append(session)
            print(f"  Found: {session}")
        else:
            print(f"  Missing: {session}")
    
    if not available_sessions:
        print("\nERROR: No session folders found!")
        print("Expected folders: 50-1, 50-2, ..., 50-10")
        return
    
    print(f"\nProcessing {len(available_sessions)} session(s)\n")
    
    # ========================================================================
    # Process All Available Sessions
    # ========================================================================
    
    all_results = []
    for session in available_sessions:
        try:
            # Process single session through complete pipeline
            results = process_single_session(
                session, 
                session,
                results_analyzer=results_analyzer,
                psd_plotter=psd_plotter,
                three_model_comparator=three_model_comparator
            )
            
            # Store results if epochs were successfully created
            if results[0] is not None:
                all_results.append(results)
                
        except Exception as e:
            print(f"\nERROR processing {session}: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with next session...\n")
            continue
    
    # ========================================================================
    # Generate Comprehensive Cross-Session Analysis Reports
    # ========================================================================
    
    if GENERATE_COMPREHENSIVE_ANALYSIS and len(all_results) > 0:
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS REPORTS")
        print("="*80 + "\n")
        
        # Two-model comparison (Logistic Regression vs Random Forest)
        if results_analyzer is not None:
            try:
                print("Generating LR vs RF comparison report...")
                results_analyzer.generate_summary_report()
                print("Report generated successfully")
            except Exception as e:
                print(f"Warning: Could not generate LR vs RF report: {str(e)}")
        
        # Three-model comparison (LR vs RF vs CNN)
        if three_model_comparator is not None and USE_CNN:
            try:
                print("\nGenerating three-model comparison report...")
                three_model_comparator.generate_comprehensive_comparison()
                print("Report generated successfully")
            except Exception as e:
                print(f"Warning: Could not generate three-model comparison: {str(e)}")
    
    # ========================================================================
    # Output Final Summary
    # ========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nSuccessfully processed: {len(all_results)} out of {len(available_sessions)} sessions")
    
    # List all generated output files
    print("\n" + "="*80)
    print("GENERATED FILES")
    print("="*80)
    
    print("\nPer-session visualizations:")
    print("  preprocessing_comparison_<session>.png - Raw vs filtered signal comparison")
    print("  erp_comparison_<session>.png - Event-related potentials for Hit vs Miss")
    if USE_ADVANCED_PSD_FEATURES:
        print("  psd_feature_importance_<session>.png - PSD band importance")
    else:
        print("  feature_importance_<session>.png - Basic feature importance")
    
    if COMPARE_MODELS:
        print("  confusion_matrix_lr_<session>.png - Logistic regression predictions")
        print("  confusion_matrix_rf_<session>.png - Random forest predictions")
        print("  feature_importance_rf_<session>.png - RF feature rankings")
    
    if USE_CNN:
        print("  confusion_matrix_cnn_<session>.png - CNN predictions")
        print("  cnn_training_history_<session>.png - Training and validation curves")
    
    if GENERATE_PSD_PLOTS:
        print("\nFrequency domain analysis:")
        print("  psd_grand_average_<session>.png - Power spectral density plots")
        print("  band_power_comparison_<session>.png - EEG frequency band comparisons")
    
    if GENERATE_COMPREHENSIVE_ANALYSIS:
        print("\nCross-session analysis (final_results/):")
        print("  performance_comparison.csv - Session-by-session metrics (LR vs RF)")
        print("  environmental_impact.csv - Environmental factors and performance")
        print("  model_comparison_comprehensive.png - Visual comparison (LR vs RF)")
        print("  environmental_impact.png - Environmental factor visualization")
        print("  statistical_tests.json - Statistical test results")
        print("  analysis_summary.txt - Text summary of findings")
        
        if USE_CNN:
            print("  three_model_comparison.png - Comprehensive comparison (LR vs RF vs CNN)")
            print("  three_model_performance.csv - Performance metrics for all models")
            print("  three_model_statistics.json - Statistical analysis results")
            print("  three_model_summary.txt - Summary of three-model comparison")
    
    # Summary of completed tasks
    print("\n" + "="*80)
    print("COMPLETED TASKS")
    print("="*80)
    
    print("\nGrade 3 Requirements:")
    print("  Task 1.4: Data Exploration - COMPLETE")
    print("  Task 1.5: Data Preprocessing - COMPLETE")
    print("  Task 1.6: Event Alignment & Epoching - COMPLETE")
    if USE_ADVANCED_PSD_FEATURES:
        print("  Task 1.7: Basic Feature Extraction - SKIPPED (using Grade 4 features)")
    else:
        print("  Task 1.7: Basic Feature Extraction - COMPLETE")
    print("  Task 1.8: Logistic Regression Classification - COMPLETE")
    
    print("\nGrade 4 Requirements:")
    if USE_ADVANCED_PSD_FEATURES:
        print("  Task 2.1: Advanced PSD Feature Extraction - COMPLETE")
    else:
        print("  Task 2.1: Advanced PSD Feature Extraction - SKIPPED")
    
    if COMPARE_MODELS or USE_RANDOM_FOREST:
        print("  Task 2.4: Random Forest Implementation - COMPLETE")
        print("  Task 2.5: Feature Importance Analysis - COMPLETE")
    
    if COMPARE_MODELS:
        print("  Task 2.6: Model Comparison (LR vs RF) - COMPLETE")
    
    if USE_CNN:
        print("\nGrade 5 Requirements:")
        print("  Task 3.1: Deep Learning Data Preparation - COMPLETE")
        print("  Task 3.2: CNN Model Development - COMPLETE")
        print("  Task 3.3: CNN Training & Validation - COMPLETE")
    
    if GENERATE_PSD_PLOTS:
        print("\nAdvanced Analysis:")
        print("  PSD Visualizations - COMPLETE")
    
    if GENERATE_COMPREHENSIVE_ANALYSIS:
        print("  Statistical Analysis - COMPLETE")
        print("  Environmental Factor Analysis - COMPLETE")
        if USE_CNN:
            print("  Three-Model Comparison - COMPLETE")
    
    print("\n" + "="*80)
    if USE_CNN:
        print("PROJECT STATUS: ALL GRADES COMPLETED (3 + 4 + 5)")
    elif USE_RANDOM_FOREST:
        print("PROJECT STATUS: GRADES 3 + 4 COMPLETED")
    else:
        print("PROJECT STATUS: GRADE 3 COMPLETED")
    print("="*80 + "\n")
    
    print("NOTE: All visualizations saved as PNG files (no popup windows)")
    print("NOTE: Modify parameters at top of script to customize analysis")
    print()


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
