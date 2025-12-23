# ============================================================================
# TASK 1.7: BASIC FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """
    Extracts features from EEG epochs for machine learning classification.
    
    Features include:
    - Time-domain: mean amplitude, variance, peak-to-peak
    - Frequency-domain: power spectral density in different bands
    - Statistical: skewness, kurtosis
    """
    
    def __init__(self, epochs):
        """
        Initialize feature extractor with epoched data.
        
        Args:
            epochs (MNE Epochs): Preprocessed and epoched EEG data
        """
        self.epochs = epochs
        self.features = None
        self.labels = None
        self.feature_names = None
        
    def extract_features(self):
        """
        Extract features from all epochs.
        
        Returns:
            tuple: (features array, labels array, feature names list)
            
        Feature organization:
        - For each channel (8 channels):
          * Mean amplitude
          * Variance
          * Peak-to-peak amplitude
          * Skewness
          * Kurtosis
          * Delta band power (1-4 Hz)
          * Theta band power (4-8 Hz)
          * Alpha band power (8-13 Hz)
          * Beta band power (13-30 Hz)
        Total: 9 features × 8 channels = 72 features per trial
        """
        import numpy as np

        print("\n" + "="*80)
        print("TASK 1.7: EXTRACTING FEATURES")
        print("="*80)
        
        # Get epoch data and labels
        data = self.epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        labels = self.epochs.events[:, 2]  # 0=Miss, 1=Hit
        
        n_epochs = data.shape[0]
        n_channels = data.shape[1]
        sfreq = self.epochs.info['sfreq']
        ch_names = self.epochs.info['ch_names']
        
        print(f"Extracting features from {n_epochs} epochs")
        print(f"Channels: {n_channels}, Sampling rate: {sfreq} Hz")
        
        # Initialize feature list
        feature_list = []
        feature_names = []
        
        # Extract features for each epoch
        for epoch_idx in range(n_epochs):
            epoch_features = []
            
            # Process each channel
            for ch_idx in range(n_channels):
                ch_name = ch_names[ch_idx]
                signal = data[epoch_idx, ch_idx, :]
                
                # Time-domain features
                mean_amp = np.mean(signal)
                variance = np.var(signal)
                peak_to_peak = np.ptp(signal)
                
                # Statistical features
                from scipy import stats
                skewness = stats.skew(signal)
                kurt = stats.kurtosis(signal)
                
                # Frequency-domain features (Power Spectral Density)
                from scipy.signal import welch
                freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, len(signal)))
                
                # Calculate band powers
                delta_power = np.mean(psd[(freqs >= 1) & (freqs < 4)])
                theta_power = np.mean(psd[(freqs >= 4) & (freqs < 8)])
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs < 13)])
                beta_power = np.mean(psd[(freqs >= 13) & (freqs < 30)])
                
                # Append all features for this channel
                epoch_features.extend([
                    mean_amp, variance, peak_to_peak,
                    skewness, kurt,
                    delta_power, theta_power, alpha_power, beta_power
                ])
                
                # Store feature names (only for first epoch)
                if epoch_idx == 0:
                    feature_names.extend([
                        f'{ch_name}_mean', f'{ch_name}_var', f'{ch_name}_ptp',
                        f'{ch_name}_skew', f'{ch_name}_kurt',
                        f'{ch_name}_delta', f'{ch_name}_theta',
                        f'{ch_name}_alpha', f'{ch_name}_beta'
                    ])
            
            feature_list.append(epoch_features)
        
        # Convert to numpy array
        self.features = np.array(feature_list)
        self.labels = labels
        self.feature_names = feature_names
        
        print(f"\nFeature extraction complete!")
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Total features per trial: {len(feature_names)}")
        print(f"Labels: {np.sum(labels == 1)} Hits, {np.sum(labels == 0)} Misses")
        print("="*80 + "\n")
        
        return self.features, self.labels, self.feature_names
    
    def plot_feature_importance(self, top_n=20, session_name=None):
        """
        Visualize top features that differ between Hit and Miss.
        
        Args:
            top_n (int): Number of top features to display
            session_name: Name of the session
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.features is None:
            print("⚠ Extract features first using extract_features()")
            return
        
        # Calculate mean difference between Hit and Miss for each feature
        hit_mask = self.labels == 1
        miss_mask = self.labels == 0
        
        hit_means = np.mean(self.features[hit_mask], axis=0)
        miss_means = np.mean(self.features[miss_mask], axis=0)
        
        # Calculate absolute difference
        differences = np.abs(hit_means - miss_means)
        
        # Get top N features
        top_indices = np.argsort(differences)[-top_n:][::-1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(top_n)
        ax.barh(y_pos, differences[top_indices], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in top_indices])
        ax.set_xlabel('Absolute Difference (Hit - Miss)')
        
        title = f"Top {top_n} Discriminative Features"
        if session_name:
            title = f"{session_name} — {title}"
        ax.set_title(title)
        
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("Saved feature importance plot: feature_importance.png")
        plt.close()
