"""
CNN EEG Classifier Module
Tasks 3.2 & 3.3: CNN Model Development, Training & Validation

Student: Studenka Lundahl
Supervisor: Adoul Mohammed Amin
Date: December 14, 2025

Description:
This module implements a 1D Convolutional Neural Network (CNN) for EEG-based
classification of golf putting performance (Hit vs Miss).

Architecture:
- Input: (n_channels=8, n_timepoints=1500) 3D EEG temporal data
- Multiple Conv1D layers for temporal feature extraction
- MaxPooling for dimensionality reduction
- Dropout for regularization
- Dense layers for classification
- Output: Binary classification (Hit/Miss)

Key Features:
- Automatic model building with configurable architecture
- Training with early stopping and learning rate scheduling
- K-fold cross-validation
- Comprehensive evaluation metrics
- Confusion matrix visualization
- Training history plots
- Model saving/loading
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, 
                                         Dropout, BatchNormalization, GlobalAveragePooling1D)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not available. CNN functionality disabled.")


class CNNEEGClassifier:
    """
    1D Convolutional Neural Network for EEG classification.
    
    This class implements a CNN architecture suitable for temporal EEG data.
    The model learns to extract relevant temporal patterns from multi-channel
    EEG signals for performance prediction.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, 
                 n_channels=8, n_timepoints=1500, verbose=True):
        """
        Initialize the CNN classifier.
        
        Args:
            X_train (np.ndarray): Training data (n_train, n_channels, n_timepoints)
            X_test (np.ndarray): Test data (n_test, n_channels, n_timepoints)
            y_train (np.ndarray): Training labels (n_train,)
            y_test (np.ndarray): Test labels (n_test,)
            n_channels (int): Number of EEG channels
            n_timepoints (int): Number of timepoints per epoch
            verbose (bool): Print detailed information
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for CNN. Install: pip install tensorflow")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.verbose = verbose
        
        self.model = None
        self.history = None
        self.y_pred = None
        self.y_pred_proba = None
        
        if self.verbose:
            print("CNNEEGClassifier initialized")
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Test samples: {X_test.shape[0]}")
            print(f"Input shape: ({n_channels} channels, {n_timepoints} timepoints)")
            print(f"Class distribution (train): Hit={np.sum(y_train==0)}, Miss={np.sum(y_train==1)}")
    
    def build_model(self, architecture='standard'):
        """
        Build the CNN model architecture.
        
        Args:
            architecture (str): Model architecture variant
                - 'standard': Balanced model with 3 conv blocks
                - 'deep': Deeper model with 4 conv blocks
                - 'shallow': Lightweight model with 2 conv blocks
                
        Returns:
            keras.Model: Compiled CNN model
        """
        if self.verbose:
            print(f"\nBuilding CNN model (architecture: {architecture})...")
        
        model = Sequential(name=f'EEG_CNN_{architecture}')
        
        # Input shape: (n_channels, n_timepoints)
        # For Conv1D, we treat channels as the sequence and timepoints as features
        # So we need to transpose: (n_timepoints, n_channels)
        
        if architecture == 'shallow':
            # Lightweight architecture (faster training, fewer parameters)
            model.add(Conv1D(filters=32, kernel_size=50, activation='relu', 
                           input_shape=(self.n_timepoints, self.n_channels), 
                           padding='same', name='conv1'))
            model.add(BatchNormalization(name='bn1'))
            model.add(MaxPooling1D(pool_size=4, name='pool1'))
            model.add(Dropout(0.3, name='dropout1'))
            
            model.add(Conv1D(filters=64, kernel_size=25, activation='relu', 
                           padding='same', name='conv2'))
            model.add(BatchNormalization(name='bn2'))
            model.add(MaxPooling1D(pool_size=4, name='pool2'))
            model.add(Dropout(0.3, name='dropout2'))
            
            model.add(GlobalAveragePooling1D(name='gap'))
            model.add(Dense(64, activation='relu', name='dense1'))
            model.add(Dropout(0.4, name='dropout3'))
            
        elif architecture == 'deep':
            # Deep architecture (more parameters, potentially better performance)
            model.add(Conv1D(filters=32, kernel_size=50, activation='relu', 
                           input_shape=(self.n_timepoints, self.n_channels), 
                           padding='same', name='conv1'))
            model.add(BatchNormalization(name='bn1'))
            model.add(MaxPooling1D(pool_size=4, name='pool1'))
            model.add(Dropout(0.2, name='dropout1'))
            
            model.add(Conv1D(filters=64, kernel_size=25, activation='relu', 
                           padding='same', name='conv2'))
            model.add(BatchNormalization(name='bn2'))
            model.add(MaxPooling1D(pool_size=4, name='pool2'))
            model.add(Dropout(0.3, name='dropout2'))
            
            model.add(Conv1D(filters=128, kernel_size=10, activation='relu', 
                           padding='same', name='conv3'))
            model.add(BatchNormalization(name='bn3'))
            model.add(MaxPooling1D(pool_size=2, name='pool3'))
            model.add(Dropout(0.3, name='dropout3'))
            
            model.add(Conv1D(filters=256, kernel_size=5, activation='relu', 
                           padding='same', name='conv4'))
            model.add(BatchNormalization(name='bn4'))
            model.add(GlobalAveragePooling1D(name='gap'))
            
            model.add(Dense(128, activation='relu', name='dense1'))
            model.add(Dropout(0.4, name='dropout4'))
            model.add(Dense(64, activation='relu', name='dense2'))
            model.add(Dropout(0.4, name='dropout5'))
            
        else:  # 'standard' architecture (recommended)
            # Balanced architecture (good performance/speed tradeoff)
            model.add(Conv1D(filters=32, kernel_size=50, activation='relu', 
                           input_shape=(self.n_timepoints, self.n_channels), 
                           padding='same', name='conv1'))
            model.add(BatchNormalization(name='bn1'))
            model.add(MaxPooling1D(pool_size=4, name='pool1'))
            model.add(Dropout(0.3, name='dropout1'))
            
            model.add(Conv1D(filters=64, kernel_size=25, activation='relu', 
                           padding='same', name='conv2'))
            model.add(BatchNormalization(name='bn2'))
            model.add(MaxPooling1D(pool_size=4, name='pool2'))
            model.add(Dropout(0.3, name='dropout2'))
            
            model.add(Conv1D(filters=128, kernel_size=10, activation='relu', 
                           padding='same', name='conv3'))
            model.add(BatchNormalization(name='bn3'))
            model.add(MaxPooling1D(pool_size=2, name='pool3'))
            model.add(Dropout(0.4, name='dropout3'))
            
            model.add(GlobalAveragePooling1D(name='gap'))
            model.add(Dense(128, activation='relu', name='dense1'))
            model.add(Dropout(0.5, name='dropout4'))
            model.add(Dense(64, activation='relu', name='dense2'))
            model.add(Dropout(0.5, name='dropout5'))
        
        # Output layer (binary classification)
        model.add(Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        if self.verbose:
            print("\nModel Architecture:")
            model.summary()
            print(f"\nTotal parameters: {model.count_params():,}")
        
        return model
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.2,
                   early_stopping_patience=15, reduce_lr_patience=7):
        """
        Train the CNN model with callbacks.
        
        Args:
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data for validation
            early_stopping_patience (int): Epochs to wait before early stopping
            reduce_lr_patience (int): Epochs to wait before reducing learning rate
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model()
        
        if self.verbose:
            print("\n=== Training CNN Model ===\n")
            print(f"Epochs: {epochs}")
            print(f"Batch size: {batch_size}")
            print(f"Validation split: {validation_split}")
        
        # Transpose data: (n_samples, n_channels, n_timepoints) -> (n_samples, n_timepoints, n_channels)
        X_train_transposed = np.transpose(self.X_train, (0, 2, 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1 if self.verbose else 0
            )
        ]
        
        # Class weights for imbalanced data
        n_class_0 = np.sum(self.y_train == 0)
        n_class_1 = np.sum(self.y_train == 1)
        total = len(self.y_train)
        
        class_weight = {
            0: total / (2 * n_class_0),
            1: total / (2 * n_class_1)
        }
        
        if self.verbose:
            print(f"\nClass weights: {class_weight}")
        
        # Train model
        history = self.model.fit(
            X_train_transposed, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2 if self.verbose else 0
        )
        
        self.history = history
        
        if self.verbose:
            print("\nTraining completed!")
            print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
            print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return history
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        if self.verbose:
            print("\n=== Evaluating CNN Model on Test Set ===\n")
        
        # Transpose test data
        X_test_transposed = np.transpose(self.X_test, (0, 2, 1))
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_transposed, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # Get loss and accuracy from model
        test_loss, test_acc = self.model.evaluate(X_test_transposed, self.y_test, verbose=0)
        
        metrics = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        if self.verbose:
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['Hit', 'Miss']))
        
        return metrics
    
    def cross_validate(self, X_full, y_full, cv=5, epochs=50, batch_size=32):
        """
        Perform k-fold cross-validation.
        
        Args:
            X_full (np.ndarray): Full dataset (n_samples, n_channels, n_timepoints)
            y_full (np.ndarray): Full labels (n_samples,)
            cv (int): Number of folds
            epochs (int): Epochs per fold
            batch_size (int): Batch size
            
        Returns:
            dict: Cross-validation results
        """
        if self.verbose:
            print(f"\n=== {cv}-Fold Cross-Validation ===\n")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = []
        fold = 1
        
        for train_idx, val_idx in skf.split(X_full, y_full):
            if self.verbose:
                print(f"\nFold {fold}/{cv}:")
            
            # Split data
            X_train_fold = X_full[train_idx]
            X_val_fold = X_full[val_idx]
            y_train_fold = y_full[train_idx]
            y_val_fold = y_full[val_idx]
            
            # Transpose data
            X_train_t = np.transpose(X_train_fold, (0, 2, 1))
            X_val_t = np.transpose(X_val_fold, (0, 2, 1))
            
            # Build fresh model
            model = self.build_model_simple()
            
            # Class weights
            n_class_0 = np.sum(y_train_fold == 0)
            n_class_1 = np.sum(y_train_fold == 1)
            total = len(y_train_fold)
            class_weight = {
                0: total / (2 * n_class_0),
                1: total / (2 * n_class_1)
            }
            
            # Train
            history = model.fit(
                X_train_t, y_train_fold,
                validation_data=(X_val_t, y_val_fold),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=class_weight,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val_t, y_val_fold, verbose=0)
            cv_scores.append(val_acc)
            
            if self.verbose:
                print(f"  Validation Accuracy: {val_acc:.4f}")
            
            fold += 1
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
        
        if self.verbose:
            print(f"\n{cv}-Fold CV Results:")
            print(f"  Mean Accuracy: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
            print(f"  All fold scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        return results
    
    def build_model_simple(self):
        """Build a simple model for cross-validation (no verbosity)."""
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=50, activation='relu', 
                       input_shape=(self.n_timepoints, self.n_channels), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.3))
        
        model.add(Conv1D(filters=64, kernel_size=25, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.3))
        
        model.add(Conv1D(filters=128, kernel_size=10, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def plot_training_history(self, session_name=''):
        """
        Plot training and validation accuracy/loss curves.
        
        Args:
            session_name (str): Session identifier for filename
        """
        if self.history is None:
            print("No training history available. Train model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if session_name:
            fig.suptitle(f'Session {session_name} - CNN Training History',
                     fontsize=16, fontweight='bold')
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = f'cnn_training_history_{session_name}.png' if session_name else 'cnn_training_history.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Training history plot saved: {filename}")
    
    def plot_confusion_matrix(self, metrics, session_name=''):
        """
        Plot confusion matrix for test predictions.
        
        Args:
            metrics (dict): Dictionary containing confusion matrix
            session_name (str): Session identifier for filename
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Hit', 'Miss'],
                   yticklabels=['Hit', 'Miss'],
                   annot_kws={'fontsize': 14})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        if session_name:
            title = (f'Session {session_name} - CNN Confusion Matrix\n' 
                     f'Accuracy: {metrics["test_accuracy"]:.3f}')
        else:
            title = f'CNN Confusion Matrix\nAccuracy: {metrics["test_accuracy"]:.3f}'

        plt.title(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        filename = f'confusion_matrix_cnn_{session_name}.png' if session_name else 'confusion_matrix_cnn.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Confusion matrix plot saved: {filename}")


# Example usage
if __name__ == "__main__":
    print("CNNEEGClassifier Module")
    print("This module requires TensorFlow/Keras and prepared 3D EEG data.")
    print("\nExample usage:")
    print("  from CNNEEGClassifier import CNNEEGClassifier")
    print("  cnn = CNNEEGClassifier(X_train, X_test, y_train, y_test)")
    print("  cnn.build_model(architecture='standard')")
    print("  cnn.train_model(epochs=100, batch_size=32)")
    print("  metrics = cnn.evaluate_model()")
    print("  cnn.plot_training_history()")
    print("  cnn.plot_confusion_matrix(metrics)")
