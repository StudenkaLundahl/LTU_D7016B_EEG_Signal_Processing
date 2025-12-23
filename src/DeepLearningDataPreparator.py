"""
Deep Learning Data Preparation Module
Task 3.1: Data Preparation for Deep Learning

Student: Studenka Lundahl
Supervisor: Adoul Mohammed Amin
Date: December 14, 2025

Description:
This module prepares EEG epochs for deep learning models (CNN and LSTM).
Note: In this project, only CNN was implemented due to time constraints.
LSTM architecture was planned but not completed.
It reshapes data from 2D feature vectors to 3D temporal sequences and provides
proper train/test splitting with normalization.

Key Features:
- Reshape epochs to (n_samples, n_channels, n_timepoints) for CNN/LSTM
- Per-channel z-score normalization
- Stratified train/test splitting
- Optional data augmentation (temporal jittering, noise injection)
- Data format compatible with both 1D CNN and LSTM architectures
  (though only CNN is implemented in this project)
"""

import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DeepLearningDataPreparator:
    """
    Prepare EEG epochs for deep learning models.

    This class handles the conversion from MNE epochs to 3D arrays suitable
    for CNN and LSTM models, along with proper normalization and splitting.

    Note: This preparator outputs data in the correct 3D format for both
    architectures. The current project implements only CNN classification;
    LSTM was planned for future work but not completed due to time constraints.
    """
    
    def __init__(self, epochs, verbose=True):
        """
        Initialize the data preparator.
        
        Args:
            epochs (mne.Epochs): MNE epochs object with 'Hit'/'Miss' events
            verbose (bool): Print detailed information
        """
        self.epochs = epochs
        self.verbose = verbose
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.n_channels = None
        self.n_timepoints = None
        self.sfreq = None
        
        if self.verbose:
            print("DeepLearningDataPreparator initialized")
            print(f"Total epochs: {len(epochs)}")
            print(f"Channels: {epochs.ch_names}")
            print(f"Sampling rate: {epochs.info['sfreq']} Hz")
            print(f"Time points per epoch: {len(epochs.times)}")
    
    def prepare_data_3d(self, test_size=0.2, random_state=42, normalize=True):
        """
        Prepare 3D data for deep learning models (CNN and potentially LSTM).

        This method extracts the raw epoch data and reshapes it to 3D format:
        (n_samples, n_channels, n_timepoints)

        The 3D format is compatible with both 1D CNN (implemented) and LSTM
        (not implemented in this project) architectures.
        
        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility
            normalize (bool): Apply per-channel z-score normalization
            
        Returns:
            dict: Dictionary containing:
                - 'X_train': Training data (n_train, n_channels, n_timepoints)
                - 'X_test': Test data (n_test, n_channels, n_timepoints)
                - 'y_train': Training labels (n_train,)
                - 'y_test': Test labels (n_test,)
                - 'n_channels': Number of EEG channels
                - 'n_timepoints': Number of time points per epoch
                - 'sfreq': Sampling frequency
        """
        if self.verbose:
            print("\n=== Preparing 3D Data for Deep Learning ===\n")
        
        # Extract data from epochs: shape will be (n_epochs, n_channels, n_times)
        X = self.epochs.get_data()  # Get raw epoch data
        
        # Extract labels (0 for Hit, 1 for Miss)
        y = self.epochs.events[:, -1]
        y = np.where(y == self.epochs.event_id['Hit'], 0, 1)
        
        # Store dimensions
        self.n_channels = X.shape[1]
        self.n_timepoints = X.shape[2]
        self.sfreq = self.epochs.info['sfreq']
        
        if self.verbose:
            print(f"Data shape: {X.shape}")
            print(f"  - Samples: {X.shape[0]}")
            print(f"  - Channels: {X.shape[1]}")
            print(f"  - Timepoints: {X.shape[2]}")
            print(f"\nLabel distribution:")
            print(f"  - Hit (0): {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
            print(f"  - Miss (1): {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        if self.verbose:
            print(f"\nTrain/Test Split:")
            print(f"  - Training samples: {X_train.shape[0]}")
            print(f"  - Test samples: {X_test.shape[0]}")
            print(f"  - Train Hit/Miss: {np.sum(y_train == 0)}/{np.sum(y_train == 1)}")
            print(f"  - Test Hit/Miss: {np.sum(y_test == 0)}/{np.sum(y_test == 1)}")
        
        # Normalize data if requested
        if normalize:
            X_train, X_test = self._normalize_data(X_train, X_test)
        
        # Store results
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_channels': self.n_channels,
            'n_timepoints': self.n_timepoints,
            'sfreq': self.sfreq
        }
    
    def _normalize_data(self, X_train, X_test):
        """
        Apply per-channel z-score normalization.
        
        Each channel is normalized independently using the training set statistics.
        This prevents data leakage from test to train.
        
        Args:
            X_train (np.ndarray): Training data (n_train, n_channels, n_timepoints)
            X_test (np.ndarray): Test data (n_test, n_channels, n_timepoints)
            
        Returns:
            tuple: (X_train_normalized, X_test_normalized)
        """
        if self.verbose:
            print("\nApplying per-channel z-score normalization...")
        
        # Store original shapes
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_channels = X_train.shape[1]
        n_timepoints = X_train.shape[2]
        
        # Initialize normalized arrays
        X_train_norm = np.zeros_like(X_train)
        X_test_norm = np.zeros_like(X_test)
        
        # Normalize each channel independently
        for ch in range(n_channels):
            # Get data for this channel: (n_samples, n_timepoints)
            train_ch = X_train[:, ch, :]
            test_ch = X_test[:, ch, :]
            
            # Reshape to 2D for StandardScaler: (n_samples, n_timepoints)
            # Note: We could also normalize across all timepoints, but per-timepoint
            # normalization is more common for EEG
            
            # Compute statistics from training data only
            mean = np.mean(train_ch)
            std = np.std(train_ch)
            
            # Avoid division by zero
            if std < 1e-10:
                std = 1.0
            
            # Apply normalization
            X_train_norm[:, ch, :] = (train_ch - mean) / std
            X_test_norm[:, ch, :] = (test_ch - mean) / std
        
        if self.verbose:
            print(f"Normalization complete:")
            print(f"  - Training data: mean={np.mean(X_train_norm):.3f}, std={np.std(X_train_norm):.3f}")
            print(f"  - Test data: mean={np.mean(X_test_norm):.3f}, std={np.std(X_test_norm):.3f}")
        
        return X_train_norm, X_test_norm
    
    def augment_data(self, X, y, n_augmentations=2, noise_level=0.1, time_jitter=0.05):
        """
        Apply data augmentation techniques to increase training samples.
        
        This is optional but can help with small datasets. Augmentation includes:
        1. Adding Gaussian noise to the signal
        2. Temporal jittering (slight time shifts)
        
        Args:
            X (np.ndarray): Data to augment (n_samples, n_channels, n_timepoints)
            y (np.ndarray): Labels (n_samples,)
            n_augmentations (int): Number of augmented copies per sample
            noise_level (float): Standard deviation of Gaussian noise
            time_jitter (float): Maximum fraction of signal to shift
            
        Returns:
            tuple: (X_augmented, y_augmented) with original + augmented data
        """
        if self.verbose:
            print(f"\nApplying data augmentation...")
            print(f"  - Original samples: {X.shape[0]}")
            print(f"  - Augmentations per sample: {n_augmentations}")
        
        X_aug_list = [X]  # Start with original data
        y_aug_list = [y]
        
        for aug_idx in range(n_augmentations):
            # Create augmented copy
            X_aug = X.copy()
            
            # 1. Add Gaussian noise
            noise = np.random.normal(0, noise_level, X_aug.shape)
            X_aug = X_aug + noise
            
            # 2. Apply temporal jittering (slight time shift)
            max_shift = int(X.shape[2] * time_jitter)
            if max_shift > 0:
                for i in range(X_aug.shape[0]):
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    X_aug[i] = np.roll(X_aug[i], shift, axis=1)
            
            X_aug_list.append(X_aug)
            y_aug_list.append(y)
        
        # Concatenate all augmented data
        X_final = np.concatenate(X_aug_list, axis=0)
        y_final = np.concatenate(y_aug_list, axis=0)
        
        if self.verbose:
            print(f"  - Final samples: {X_final.shape[0]}")
            print(f"  - Augmentation ratio: {X_final.shape[0] / X.shape[0]:.1f}x")
        
        return X_final, y_final
    
    def get_data_info(self):
        """
        Get information about the prepared data.
        
        Returns:
            dict: Dictionary with data dimensions and properties
        """
        if self.X_train is None:
            return {"error": "Data not prepared yet. Call prepare_data_3d() first."}
        
        return {
            'n_train_samples': self.X_train.shape[0],
            'n_test_samples': self.X_test.shape[0],
            'n_channels': self.n_channels,
            'n_timepoints': self.n_timepoints,
            'sfreq': self.sfreq,
            'duration_seconds': self.n_timepoints / self.sfreq,
            'train_class_distribution': {
                'Hit': int(np.sum(self.y_train == 0)),
                'Miss': int(np.sum(self.y_train == 1))
            },
            'test_class_distribution': {
                'Hit': int(np.sum(self.y_test == 0)),
                'Miss': int(np.sum(self.y_test == 1))
            }
        }
    
    def print_data_summary(self):
        """Print a comprehensive summary of the prepared data."""
        info = self.get_data_info()
        
        if 'error' in info:
            print(info['error'])
            return
        
        print("\n" + "="*70)
        print("DEEP LEARNING DATA SUMMARY")
        print("="*70)
        print(f"\nData Dimensions:")
        print(f"  - Training samples: {info['n_train_samples']}")
        print(f"  - Test samples: {info['n_test_samples']}")
        print(f"  - Channels: {info['n_channels']}")
        print(f"  - Timepoints per epoch: {info['n_timepoints']}")
        print(f"  - Epoch duration: {info['duration_seconds']:.2f} seconds")
        print(f"  - Sampling frequency: {info['sfreq']:.0f} Hz")
        
        print(f"\nClass Distribution (Training):")
        print(f"  - Hit: {info['train_class_distribution']['Hit']} "
              f"({info['train_class_distribution']['Hit']/info['n_train_samples']*100:.1f}%)")
        print(f"  - Miss: {info['train_class_distribution']['Miss']} "
              f"({info['train_class_distribution']['Miss']/info['n_train_samples']*100:.1f}%)")
        
        print(f"\nClass Distribution (Test):")
        print(f"  - Hit: {info['test_class_distribution']['Hit']} "
              f"({info['test_class_distribution']['Hit']/info['n_test_samples']*100:.1f}%)")
        print(f"  - Miss: {info['test_class_distribution']['Miss']} "
              f"({info['test_class_distribution']['Miss']/info['n_test_samples']*100:.1f}%)")
        
        print("="*70 + "\n")


# Example usage (for testing this module independently)
if __name__ == "__main__":
    print("DeepLearningDataPreparator Module")
    print("This module should be imported and used with MNE epochs.")
    print("\nExample usage:")
    print("  from DeepLearningDataPreparator import DeepLearningDataPreparator")
    print("  preparator = DeepLearningDataPreparator(epochs)")
    print("  data = preparator.prepare_data_3d(test_size=0.2)")
    print("  preparator.print_data_summary()")
