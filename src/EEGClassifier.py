# ============================================================================
# TASK 1.8: LOGISTIC REGRESSION IMPLEMENTATION
# ============================================================================

class EEGClassifier:
    """
    Classifies EEG trials as Hit or Miss using Logistic Regression.
    
    Implements proper train-test split and cross-validation for
    robust performance evaluation.
    """
    
    def __init__(self, features, labels, feature_names=None):
        """
        Initialize classifier with feature data.
        
        Args:
            features (numpy array): Feature matrix (n_samples, n_features)
            labels (numpy array): Labels (0=Miss, 1=Hit)
            feature_names (list): Names of features (optional)
        """
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing (0-1)
            random_state (int): Random seed for reproducibility
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        print("\n" + "="*80)
        print("TASK 1.8: PREPARING DATA FOR CLASSIFICATION")
        print("="*80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels  # Maintain class proportions
        )
        
        # Standardize features (zero mean, unit variance)
        # Important for logistic regression convergence
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"  Hits: {np.sum(self.y_train == 1)}, Misses: {np.sum(self.y_train == 0)}")
        print(f"Test set: {len(self.X_test)} samples")
        print(f"  Hits: {np.sum(self.y_test == 1)}, Misses: {np.sum(self.y_test == 0)}")
        print("="*80 + "\n")
    
    def train_model(self):
        """
        Train Logistic Regression model.
        
        Returns:
            Trained model
        """
        from sklearn.linear_model import LogisticRegression
        
        print("\n" + "="*80)
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print("="*80)
        
        # Initialize and train model
        self.model = LogisticRegression(
            max_iter=1000,  # Increase iterations for convergence
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training complete!")
        print("="*80 + "\n")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate model performance on test set.
        
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report
        
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Make predictions
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        precision = precision_score(self.y_test, y_pred_test)
        recall = recall_score(self.y_test, y_pred_test)
        f1 = f1_score(self.y_test, y_pred_test)
        
        print(f"\nPerformance Metrics:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy:     {test_acc:.3f}")
        print(f"  Precision:         {precision:.3f}")
        print(f"  Recall:            {recall:.3f}")
        print(f"  F1-Score:          {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (Miss):  {cm[0, 0]}")
        print(f"  False Positives:        {cm[0, 1]}")
        print(f"  False Negatives:        {cm[1, 0]}")
        print(f"  True Positives (Hit):   {cm[1, 1]}")
        
        # Detailed report
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred_test, 
                                   target_names=['Miss', 'Hit']))
        
        print("="*80 + "\n")
        
        # Return metrics dictionary
        metrics = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def cross_validate(self, cv=5):
        """
        Perform cross-validation for more robust evaluation.
        
        Args:
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        
        print("\n" + "="*80)
        print("CROSS-VALIDATION ANALYSIS")
        print("="*80)
        print(f"Performing {cv}-fold cross-validation...")
        
        # Create model
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, self.features, self.labels, cv=cv, scoring='accuracy')
        
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print("="*80 + "\n")
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
    
    def plot_confusion_matrix(self, metrics, session_name=None):
        """
        Visualize confusion matrix.
        
        Args:
            metrics (dict): Dictionary containing confusion matrix
            session_name (str): Name of the session for plot title
        """
        import matplotlib.pyplot as plt
        import numpy as np

        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Labels
        title = "Logistic Regression - Confusion Matrix"
        if session_name:
            title = f"{session_name} — {title}"
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Miss', 'Hit'],
               yticklabels=['Miss', 'Hit'],
               ylabel='True Label',
               xlabel='Predicted Label',
               title=title)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=20)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("Saved confusion matrix plot: confusion_matrix.png")
        plt.close()
