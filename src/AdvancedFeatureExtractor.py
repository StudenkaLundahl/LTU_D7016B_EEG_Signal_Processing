# ============================================================================
# TASK 2.1: ADVANCED FEATURE EXTRACTION - POWER SPECTRAL DENSITY (PSD)
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis

class AdvancedFeatureExtractor:
    """
    Extracts advanced features from EEG epochs for improved classification.
    
    This class extends basic feature extraction by computing:
    - Detailed Power Spectral Density (PSD) across multiple frequency bands
    - Band power ratios (e.g., theta/beta ratio for attention)
    - Relative band powers (normalized by total power)
    - Enhanced frequency-domain features
    
    The PSD approach provides better frequency resolution and captures
    more nuanced patterns in brain oscillations that distinguish Hit from Miss trials.
    """
    
    def __init__(self, epochs):
        """
        Initialize advanced feature extractor with epoched data.
        
        Args:
            epochs (MNE Epochs): Preprocessed and epoched EEG data
        """
        self.epochs = epochs
        self.features = None
        self.labels = None
        self.feature_names = None
        self.sfreq = epochs.info['sfreq']
        self.ch_names = epochs.info['ch_names']
        
    def extract_psd_features(self, fmin=1.0, fmax=40.0, n_fft=256):
        """
        Extract advanced Power Spectral Density (PSD) features from all epochs.
        
        Args:
            fmin (float): Minimum frequency for PSD computation (Hz)
            fmax (float): Maximum frequency for PSD computation (Hz)
            n_fft (int): FFT window length for Welch's method
            
        Returns:
            tuple: (features array, labels array, feature names list)
            
        Feature organization per channel (8 channels):
        - Absolute band powers (5 bands: Delta, Theta, Alpha, Beta, Gamma)
        - Relative band powers (5 bands: normalized by total power)
        - Band power ratios (4 ratios: Theta/Beta, Alpha/Theta, Beta/Alpha, (Theta+Alpha)/Beta)
        - Total power across all frequencies
        - Peak frequency (frequency with maximum power)
        - Spectral edge frequency (frequency below which 95% of power is contained)
        - Power distribution metrics (mean frequency, spectral centroid)
        
        Total: 19 features × 8 channels = 152 features per trial
        """
        print("\n" + "="*80)
        print("TASK 2.1: EXTRACTING ADVANCED PSD FEATURES")
        print("="*80)
        
        # Get epoch data and labels
        data = self.epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        labels = self.epochs.events[:, 2]  # 0=Miss, 1=Hit
        
        n_epochs = data.shape[0]
        n_channels = data.shape[1]
        
        print(f"Extracting PSD features from {n_epochs} epochs")
        print(f"Channels: {n_channels}, Sampling rate: {self.sfreq} Hz")
        print(f"Frequency range: {fmin}-{fmax} Hz")
        print(f"FFT window length: {n_fft} samples")
        
        # Define frequency bands (in Hz)
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # Initialize feature list
        feature_list = []
        feature_names = []
        
        # Extract features for each epoch
        for epoch_idx in range(n_epochs):
            epoch_features = []
            
            # Process each channel
            for ch_idx in range(n_channels):
                ch_name = self.ch_names[ch_idx]
                signal = data[epoch_idx, ch_idx, :]
                
                # ================================================================
                # 1. COMPUTE POWER SPECTRAL DENSITY using Welch's method
                # ================================================================
                # Welch's method: averages PSD over overlapping segments
                # to reduce variance and provide smoother frequency estimates
                nperseg = min(n_fft, len(signal))
                freqs, psd = welch(
                    signal, 
                    fs=self.sfreq, 
                    nperseg=nperseg,
                    noverlap=nperseg // 2,  # 50% overlap
                    window='hamming'
                )
                
                # Limit to frequency range of interest
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                freqs = freqs[freq_mask]
                psd = psd[freq_mask]
                
                # ================================================================
                # 2. ABSOLUTE BAND POWERS
                # ================================================================
                # Calculate power in each frequency band
                band_powers = {}
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    band_power = np.trapz(psd[band_mask], freqs[band_mask])
                    band_powers[band_name] = band_power
                    epoch_features.append(band_power)
                
                # Store feature names (only for first epoch and first channel)
                if epoch_idx == 0 and ch_idx == 0:
                    for band_name in bands.keys():
                        feature_names.append(f'{ch_name}_abs_{band_name}')
                elif epoch_idx == 0:
                    # For other channels in first epoch
                    for band_name in bands.keys():
                        feature_names.append(f'{ch_name}_abs_{band_name}')
                
                # ================================================================
                # 3. RELATIVE BAND POWERS (normalized by total power)
                # ================================================================
                # Relative power helps account for individual differences in
                # overall EEG amplitude
                total_power = np.trapz(psd, freqs)
                
                for band_name in bands.keys():
                    rel_power = band_powers[band_name] / (total_power + 1e-10)
                    epoch_features.append(rel_power)
                
                if epoch_idx == 0:
                    for band_name in bands.keys():
                        feature_names.append(f'{ch_name}_rel_{band_name}')
                
                # ================================================================
                # 4. BAND POWER RATIOS
                # ================================================================
                # These ratios are linked to cognitive states:
                # - Theta/Beta: attention and cognitive load
                # - Alpha/Theta: relaxation vs active thinking
                # - Beta/Alpha: mental activation
                # - (Theta+Alpha)/Beta: engagement index
                
                theta_beta_ratio = band_powers['theta'] / (band_powers['beta'] + 1e-10)
                alpha_theta_ratio = band_powers['alpha'] / (band_powers['theta'] + 1e-10)
                beta_alpha_ratio = band_powers['beta'] / (band_powers['alpha'] + 1e-10)
                engagement_idx = (band_powers['theta'] + band_powers['alpha']) / (band_powers['beta'] + 1e-10)
                
                epoch_features.extend([
                    theta_beta_ratio,
                    alpha_theta_ratio,
                    beta_alpha_ratio,
                    engagement_idx
                ])
                
                if epoch_idx == 0:
                    feature_names.extend([
                        f'{ch_name}_theta_beta_ratio',
                        f'{ch_name}_alpha_theta_ratio',
                        f'{ch_name}_beta_alpha_ratio',
                        f'{ch_name}_engagement_idx'
                    ])
                
                # ================================================================
                # 5. GLOBAL POWER METRICS
                # ================================================================
                epoch_features.append(total_power)
                
                if epoch_idx == 0:
                    feature_names.append(f'{ch_name}_total_power')
                
                # ================================================================
                # 6. PEAK FREQUENCY
                # ================================================================
                # Frequency with maximum power - indicates dominant oscillation
                peak_freq = freqs[np.argmax(psd)]
                epoch_features.append(peak_freq)
                
                if epoch_idx == 0:
                    feature_names.append(f'{ch_name}_peak_freq')
                
                # ================================================================
                # 7. SPECTRAL EDGE FREQUENCY (SEF95)
                # ================================================================
                # Frequency below which 95% of the power is contained
                # Useful for detecting shifts in frequency distribution
                cumsum_psd = np.cumsum(psd)
                sef95_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
                sef95 = freqs[sef95_idx[0]] if len(sef95_idx) > 0 else fmax
                epoch_features.append(sef95)
                
                if epoch_idx == 0:
                    feature_names.append(f'{ch_name}_sef95')
                
                # ================================================================
                # 8. SPECTRAL MOMENTS
                # ================================================================
                # Mean frequency (spectral centroid): weighted average frequency
                mean_freq = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
                
                # Spectral spread: weighted standard deviation of frequencies
                spectral_spread = np.sqrt(
                    np.sum(((freqs - mean_freq) ** 2) * psd) / (np.sum(psd) + 1e-10)
                )
                
                epoch_features.extend([mean_freq, spectral_spread])
                
                if epoch_idx == 0:
                    feature_names.extend([
                        f'{ch_name}_mean_freq',
                        f'{ch_name}_spectral_spread'
                    ])
            
            feature_list.append(epoch_features)
        
        # Convert to numpy array
        self.features = np.array(feature_list)
        self.labels = labels
        self.feature_names = feature_names
        
        print(f"\nPSD feature extraction complete!")
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Total features per trial: {len(feature_names)}")
        print(f"Features should match: {self.features.shape[1]} == {len(feature_names)}")
        print(f"Labels: {np.sum(labels == 1)} Hits, {np.sum(labels == 0)} Misses")
        
        # Verify feature count matches expected
        expected_features = 19 * n_channels  # 19 features per channel
        if len(feature_names) != expected_features:
            print(f"\n⚠ WARNING: Feature count mismatch!")
            print(f"Expected: {expected_features} features")
            print(f"Got: {len(feature_names)} feature names")
        else:
            print(f"\n✓ Feature count verified: {expected_features} features")
        
        # Print feature breakdown
        print(f"\nFeature breakdown per channel:")
        print(f"  - Absolute band powers: 5")
        print(f"  - Relative band powers: 5")
        print(f"  - Band power ratios: 4")
        print(f"  - Total power: 1")
        print(f"  - Peak frequency: 1")
        print(f"  - Spectral edge freq (SEF95): 1")
        print(f"  - Spectral moments: 2")
        print(f"  Total per channel: 19")
        print(f"  Total for {n_channels} channels: {len(feature_names)}")
        print("="*80 + "\n")
        
        return self.features, self.labels, self.feature_names
    
    def plot_feature_importance(self, top_n=20, session_name=None):
        """
        Visualize top PSD features that differ between Hit and Miss.
        
        Args:
            top_n (int): Number of top features to display
            session_name (str): Name of the session for plot title
        """
        if self.features is None:
            print("⚠ Extract features first using extract_psd_features()")
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
        
        ax.barh(y_pos, differences[top_indices], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in top_indices], fontsize=9)
        ax.set_xlabel('Absolute Difference (Hit - Miss)', fontsize=11)
        
        title = f"Top {top_n} Discriminative PSD Features"
        if session_name:
            title = f"{session_name} — {title}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add legend for color coding
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
        plt.savefig('psd_feature_importance.png', dpi=150, bbox_inches='tight')
        print("Saved PSD feature importance plot: psd_feature_importance.png")
        plt.close()
    
    def plot_psd_comparison(self, channel_idx=0, session_name=None):
        """
        Plot average PSD comparison between Hit and Miss trials for a specific channel.
        
        Args:
            channel_idx (int): Index of channel to plot (0-7)
            session_name (str): Name of the session for plot title
        """
        # Get epoch data
        data = self.epochs.get_data()
        labels = self.epochs.events[:, 2]
        
        # Separate Hit and Miss trials
        hit_data = data[labels == 1, channel_idx, :]
        miss_data = data[labels == 0, channel_idx, :]
        
        # Compute average PSD for each condition
        freqs_hit, psd_hit = welch(
            hit_data.T, 
            fs=self.sfreq, 
            nperseg=min(256, hit_data.shape[1]),
            axis=0
        )
        psd_hit_mean = np.mean(psd_hit, axis=1)
        psd_hit_std = np.std(psd_hit, axis=1)
        
        freqs_miss, psd_miss = welch(
            miss_data.T, 
            fs=self.sfreq, 
            nperseg=min(256, miss_data.shape[1]),
            axis=0
        )
        psd_miss_mean = np.mean(psd_miss, axis=1)
        psd_miss_std = np.std(psd_miss, axis=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot with shaded error regions
        ax.plot(freqs_hit, psd_hit_mean, 'b-', linewidth=2, label='Hit', alpha=0.8)
        ax.fill_between(
            freqs_hit, 
            psd_hit_mean - psd_hit_std, 
            psd_hit_mean + psd_hit_std,
            color='blue', alpha=0.2
        )
        
        ax.plot(freqs_miss, psd_miss_mean, 'r-', linewidth=2, label='Miss', alpha=0.8)
        ax.fill_between(
            freqs_miss, 
            psd_miss_mean - psd_miss_std, 
            psd_miss_mean + psd_miss_std,
            color='red', alpha=0.2
        )
        
        # Add frequency band regions
        bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 
                 'Beta': (13, 30), 'Gamma': (30, 40)}
        colors_band = ['lightgray', 'lightyellow', 'lightgreen', 'lightblue', 'lightcoral']
        
        y_max = max(np.max(psd_hit_mean), np.max(psd_miss_mean))
        for (band_name, (low, high)), color in zip(bands.items(), colors_band):
            ax.axvspan(low, high, alpha=0.1, color=color, label=band_name)
        
        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11)
        ax.set_xlim(1, 40)
        ax.set_yscale('log')
        
        ch_name = self.ch_names[channel_idx]
        title = f"PSD Comparison: Hit vs Miss - Channel {ch_name}"
        if session_name:
            title = f"{session_name} — {title}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Organize legend
        handles, labels_legend = ax.get_legend_handles_labels()
        # Put Hit/Miss first, then bands
        order = [0, 1] + list(range(2, len(handles)))
        ax.legend([handles[i] for i in order], [labels_legend[i] for i in order],
                 loc='upper right', fontsize=9, ncol=2)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('psd_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved PSD comparison plot for {ch_name}: psd_comparison.png")
        plt.close()
