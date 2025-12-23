# ============================================================================
# TASK 2.4: RANDOM FOREST CLASSIFICATION
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

class RandomForestEEGClassifier:
    """
    Random Forest classifier for EEG-based Hit/Miss prediction.
    
    Random Forests offer several advantages over Logistic Regression:
    - Handle non-linear relationships between features and outcomes
    - Robust to overfitting through ensemble averaging
    - Provide feature importance rankings
    - No assumptions about data distribution
    - Can capture complex interaction effects
    
    This implementation includes:
    - Feature standardization
    - Hyperparameter optimization via Grid Search
    - Cross-validation for robust evaluation
    - Feature importance analysis
    - Model comparison capabilities
    """
    
    def __init__(self, features, labels, feature_names):
        """
        Initialize Random Forest classifier with EEG features.
        
        Args:
            features (np.array): Feature matrix (n_samples, n_features)
            labels (np.array): Binary labels (0=Miss, 1=Hit)
            feature_names (list): Names of features for interpretability
        """
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.model = None
        self.scaler = StandardScaler()
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        
        # Results storage
        self.best_params = None
        self.feature_importances = None
        
    def prepare_data(self, test_size=0.2, random_state=42, scale_features=True):
        """
        Split data into training and test sets, optionally scale features.
        
        Args:
            test_size (float): Proportion of data for testing (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
            scale_features (bool): Whether to standardize features (recommended for RF)
        """
        print("\n" + "="*80)
        print("TASK 2.4: PREPARING DATA FOR RANDOM FOREST")
        print("="*80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, 
            self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels  # Maintain class distribution
        )
        
        # Scale features if requested
        if scale_features:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print(f"Features standardized (mean=0, std=1)")
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
        
        # Print data split info
        train_hits = np.sum(self.y_train == 1)
        train_misses = np.sum(self.y_train == 0)
        test_hits = np.sum(self.y_test == 1)
        test_misses = np.sum(self.y_test == 0)
        
        print(f"Training set: {len(self.y_train)} samples")
        print(f"  Hits: {train_hits}, Misses: {train_misses}")
        print(f"Test set: {len(self.y_test)} samples")
        print(f"  Hits: {test_hits}, Misses: {test_misses}")
        print("="*80 + "\n")
        
    def train_model(self, optimize_hyperparameters=True, n_estimators=100, 
                   max_depth=None, min_samples_split=2, min_samples_leaf=1,
                   max_features=None, param_grid=None):
        """
        Train Random Forest model with optional hyperparameter optimization.
        
        Args:
            optimize_hyperparameters (bool): Whether to use GridSearchCV
            n_estimators (int): Number of trees (if not optimizing)
            max_depth (int): Maximum tree depth (if not optimizing)
            min_samples_split (int): Min samples to split node
            min_samples_leaf (int): Min samples in leaf node
        """
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*80)
        
        if optimize_hyperparameters:
            print("Performing hyperparameter optimization with GridSearchCV...")
            print("This may take a moment...\n")
            
            if param_grid is None:
                # Define parameter grid for Grid Search
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            
            
            # Initialize base model
            rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
            
            # Grid Search with cross-validation
            grid_search = GridSearchCV(
                rf_base, 
                param_grid, 
                cv=5,  # 5-fold cross-validation
                scoring='f1_weighted',
                n_jobs=-1,  # Use all CPU cores
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(self.X_train_scaled, self.y_train)
            
            # Store best model and parameters
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"\nBest hyperparameters found:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
            
        else:
            print(f"Training with specified hyperparameters:")
            print(f"  n_estimators: {n_estimators}")
            print(f"  max_depth: {max_depth}")
            print(f"  min_samples_split: {min_samples_split}")
            print(f"  min_samples_leaf: {min_samples_leaf}")
            
            # Train model with specified parameters
            rf_kwargs = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
            # Only include max_features if explicitly provided
            if max_features is not None:
                rf_kwargs['max_features'] = max_features

            self.model = RandomForestClassifier(**rf_kwargs)
            
            self.model.fit(self.X_train_scaled, self.y_train)
        
        # Extract feature importances
        self.feature_importances = self.model.feature_importances_
        
        print("\nModel training complete!")
        print("="*80 + "\n")
        
    def evaluate_model(self):
        """
        Evaluate model performance on test set.
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80 + "\n")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        train_acc = accuracy_score(self.y_train, y_train_pred)
        test_acc = accuracy_score(self.y_test, y_test_pred)
        precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_test_pred, average='weighted')
        f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # Store results
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_test_pred
        }
        
        # Print results
        print("Performance Metrics:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy:     {test_acc:.3f}")
        print(f"  Precision:         {precision:.3f}")
        print(f"  Recall:            {recall:.3f}")
        print(f"  F1-Score:          {f1:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (Miss):  {cm[0,0]}")
        print(f"  False Positives:        {cm[0,1]}")
        print(f"  False Negatives:        {cm[1,0]}")
        print(f"  True Positives (Hit):   {cm[1,1]}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            self.y_test, 
            y_test_pred, 
            target_names=['Miss', 'Hit'],
            zero_division=0
        ))
        
        print("="*80 + "\n")
        
        return metrics
    
    def cross_validate(self, cv=5):
        """
        Perform k-fold cross-validation for robust performance estimate.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        print("\n" + "="*80)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*80)
        print(f"Performing {cv}-fold cross-validation...\n")
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.X_train_scaled, 
            self.y_train, 
            cv=cv,
            scoring='accuracy'
        )
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean accuracy: {results['mean_score']:.3f} (+/- {results['std_score']:.3f})")
        print("="*80 + "\n")
        
        return results
    
    def plot_confusion_matrix(self, metrics, session_name=None):
        """
        Visualize confusion matrix as a heatmap.
        
        Args:
            metrics (dict): Dictionary containing confusion matrix
            session_name (str): Session identifier for plot title
        """
        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Miss', 'Hit'],
               yticklabels=['Miss', 'Hit'],
               xlabel='Predicted Label',
               ylabel='True Label')
        
        # Add title
        title = 'Random Forest - Confusion Matrix'
        if session_name:
            title = f"{session_name} — {title}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
        print("Saved confusion matrix plot: confusion_matrix_rf.png")
        plt.close()
    
    def plot_feature_importance(self, top_n=20, session_name=None):
        """
        Visualize Random Forest feature importances.
        
        Args:
            top_n (int): Number of top features to display
            session_name (str): Session identifier for plot title
        """
        if self.feature_importances is None:
            print("⚠ Train model first to get feature importances")
            return
        
        # Get top N features
        top_indices = np.argsort(self.feature_importances)[-top_n:][::-1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(top_n)
        colors = []
        for idx in top_indices:
            name = self.feature_names[idx]
            # Color-code by feature type
            if '_abs_' in name:
                colors.append('steelblue')
            elif '_rel_' in name:
                colors.append('coral')
            elif 'ratio' in name or 'engagement' in name:
                colors.append('mediumseagreen')
            else:
                colors.append('plum')
        
        ax.barh(y_pos, self.feature_importances[top_indices], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in top_indices], fontsize=9)
        ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=11)
        
        title = f"Top {top_n} Most Important Features (Random Forest)"
        if session_name:
            title = f"{session_name} — {title}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', label='Absolute Band Power'),
            Patch(facecolor='coral', label='Relative Band Power'),
            Patch(facecolor='mediumseagreen', label='Band Ratio'),
            Patch(facecolor='plum', label='Spectral Metric')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_rf.png', dpi=150, bbox_inches='tight')
        print("Saved feature importance plot: feature_importance_rf.png")
        plt.close()
