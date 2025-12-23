"""
Grand Average PSD Visualizer
Creates publication-quality PSD plots showing Hit vs Miss patterns
across all channels, similar to the example you showed
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal as sig
from scipy import stats

class GrandAveragePSDPlotter:
    """
    Creates grand average Power Spectral Density plots.
    
    Shows frequency characteristics of Hit vs Miss trials,
    demonstrating which frequency bands differ between conditions.
    Uses Welch's method for robust PSD estimation.
    """
    
    def __init__(self, sfreq=500):
        """
        Initialize PSD plotter.
        
        Args:
            sfreq (int): Sampling frequency in Hz
        """
        self.sfreq = sfreq
        self.channel_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        
        # Define frequency bands
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        # Colors for bands
        self.band_colors = {
            'delta': '#E8F5E9',
            'theta': '#C5E1A5',
            'alpha': '#FFF9C4',
            'beta': '#FFCCBC',
            'gamma': '#F8BBD0'
        }
    
    def compute_grand_average_psd(self, epochs, fmin=1.0, fmax=40.0, n_fft=256):
        """
        Compute grand average PSD for Hit and Miss conditions.
        
        Args:
            epochs (mne.Epochs): Epoched EEG data with 'Hit'/'Miss' events
            fmin (float): Minimum frequency
            fmax (float): Maximum frequency
            n_fft (int): FFT length for Welch's method
            
        Returns:
            tuple: (freqs, psd_hit, psd_miss, sem_hit, sem_miss)
        """
        # Separate Hit and Miss epochs
        hit_epochs = epochs['Hit']
        miss_epochs = epochs['Miss']
        
        print(f"\nComputing PSD using Welch's method...")
        print(f"  Hit trials: {len(hit_epochs)}")
        print(f"  Miss trials: {len(miss_epochs)}")
        print(f"  Frequency range: {fmin}-{fmax} Hz")
        print(f"  FFT length: {n_fft}")
        
        # Compute PSD for all trials
        psd_hit_list = []
        psd_miss_list = []
        
        # Get data
        hit_data = hit_epochs.get_data()  # (n_epochs, n_channels, n_times)
        miss_data = miss_epochs.get_data()
        
        # Compute PSD for each trial using Welch's method
        for trial in hit_data:
            psds_trial = []
            for ch_data in trial:
                freqs, psd = sig.welch(ch_data, fs=self.sfreq, nperseg=n_fft)
                # Select frequency range
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                psds_trial.append(psd[freq_mask])
            psd_hit_list.append(psds_trial)
        
        for trial in miss_data:
            psds_trial = []
            for ch_data in trial:
                freqs, psd = sig.welch(ch_data, fs=self.sfreq, nperseg=n_fft)
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                psds_trial.append(psd[freq_mask])
            psd_miss_list.append(psds_trial)
        
        # Get frequencies
        freqs_full, _ = sig.welch(hit_data[0, 0, :], fs=self.sfreq, nperseg=n_fft)
        freq_mask = (freqs_full >= fmin) & (freqs_full <= fmax)
        freqs = freqs_full[freq_mask]
        
        # Convert to arrays
        psd_hit = np.array(psd_hit_list)  # (n_trials, n_channels, n_freqs)
        psd_miss = np.array(psd_miss_list)
        
        # Compute mean and SEM
        psd_hit_mean = np.mean(psd_hit, axis=0)  # (n_channels, n_freqs)
        psd_miss_mean = np.mean(psd_miss, axis=0)
        
        psd_hit_sem = np.std(psd_hit, axis=0) / np.sqrt(psd_hit.shape[0])
        psd_miss_sem = np.std(psd_miss, axis=0) / np.sqrt(psd_miss.shape[0])
        
        return freqs, psd_hit_mean, psd_miss_mean, psd_hit_sem, psd_miss_sem
    
    def plot_per_channel_psd(self, epochs, session_name, fmin=1.0, fmax=40.0):
        """
        Plot PSD for each channel separately (8 subplots).
        Similar to the example plot shown.
        
        Args:
            epochs (mne.Epochs): Epoched data
            session_name (str): Session identifier
            fmin, fmax (float): Frequency range
        """
        freqs, psd_hit, psd_miss, sem_hit, sem_miss = self.compute_grand_average_psd(
            epochs, fmin, fmax
        )
        
        # Create figure with 4x2 subplots
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for ch_idx, (ax, ch_name) in enumerate(zip(axes, self.channel_names)):
            # Plot Hit condition
            ax.plot(freqs, psd_hit[ch_idx], color='#2ecc71', linewidth=2.5, 
                   label='Hit', alpha=0.9)
            ax.fill_between(freqs, 
                           psd_hit[ch_idx] - sem_hit[ch_idx],
                           psd_hit[ch_idx] + sem_hit[ch_idx],
                           color='#2ecc71', alpha=0.2)
            
            # Plot Miss condition
            ax.plot(freqs, psd_miss[ch_idx], color='#e74c3c', linewidth=2.5, 
                   label='Miss', alpha=0.9)
            ax.fill_between(freqs, 
                           psd_miss[ch_idx] - sem_miss[ch_idx],
                           psd_miss[ch_idx] + sem_miss[ch_idx],
                           color='#e74c3c', alpha=0.2)
            
            # Add frequency band shading
            for band_name, (low, high) in self.bands.items():
                if low >= fmin and high <= fmax:
                    ax.axvspan(low, high, alpha=0.1, color=self.band_colors[band_name])
            
            # Formatting
            ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Power (μV²/Hz)', fontsize=10, fontweight='bold')
            ax.set_title(f'{ch_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([fmin, fmax])
            
            # Add band labels on first subplot
            if ch_idx == 0:
                y_pos = ax.get_ylim()[1] * 0.95
                for band_name, (low, high) in self.bands.items():
                    if low >= fmin and high <= fmax:
                        ax.text((low + high) / 2, y_pos, band_name.capitalize(),
                               ha='center', va='top', fontsize=8, fontweight='bold')
        
        plt.suptitle(f'{session_name} — Average PSD per Channel: Hit vs Miss', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'psd_per_channel_{session_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved per-channel PSD plot: psd_per_channel_{session_name}.png")
        plt.close()
    
    def plot_grand_average_all_channels(self, epochs, session_name, fmin=1.0, fmax=40.0):
        """
        Plot grand average PSD across all channels.
        Shows overall frequency characteristics.
        
        Args:
            epochs (mne.Epochs): Epoched data
            session_name (str): Session identifier
            fmin, fmax (float): Frequency range
        """
        freqs, psd_hit, psd_miss, sem_hit, sem_miss = self.compute_grand_average_psd(
            epochs, fmin, fmax
        )
        
        # Average across all channels
        psd_hit_grand = np.mean(psd_hit, axis=0)
        psd_miss_grand = np.mean(psd_miss, axis=0)
        sem_hit_grand = np.sqrt(np.sum(sem_hit**2, axis=0)) / len(self.channel_names)
        sem_miss_grand = np.sqrt(np.sum(sem_miss**2, axis=0)) / len(self.channel_names)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot
        ax.plot(freqs, psd_hit_grand, color='#2ecc71', linewidth=3, 
               label='Hit (ext_1)', alpha=0.9)
        ax.fill_between(freqs, 
                       psd_hit_grand - sem_hit_grand,
                       psd_hit_grand + sem_hit_grand,
                       color='#2ecc71', alpha=0.2)
        
        ax.plot(freqs, psd_miss_grand, color='#e74c3c', linewidth=3, 
               label='Miss (ext_0)', alpha=0.9)
        ax.fill_between(freqs, 
                       psd_miss_grand - sem_miss_grand,
                       psd_miss_grand + sem_miss_grand,
                       color='#e74c3c', alpha=0.2)
        
        # Add frequency band shading and labels
        for band_name, (low, high) in self.bands.items():
            if low >= fmin and high <= fmax:
                ax.axvspan(low, high, alpha=0.15, color=self.band_colors[band_name],
                          label=f'{band_name.capitalize()} ({low}-{high} Hz)')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=13, fontweight='bold')
        ax.set_title(f'{session_name} — Grand Average PSD (All Channels)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([fmin, fmax])
        
        plt.tight_layout()
        plt.savefig(f'psd_grand_average_{session_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved grand average PSD plot: psd_grand_average_{session_name}.png")
        plt.close()
    
    def plot_band_power_comparison(self, epochs, session_name):
        """
        Plot bar chart comparing average band power for Hit vs Miss.
        Shows which frequency bands discriminate performance.
        Similar to the example image with theta/alpha/beta bars.
        
        Args:
            epochs (mne.Epochs): Epoched data
            session_name (str): Session identifier
        """
        # Compute band powers
        hit_band_powers = {band: [] for band in self.bands.keys()}
        miss_band_powers = {band: [] for band in self.bands.keys()}
        
        hit_epochs = epochs['Hit']
        miss_epochs = epochs['Miss']
        
        # For each channel
        for ch_idx in range(len(self.channel_names)):
            # Hit trials
            hit_data = hit_epochs.get_data()[:, ch_idx, :]  # (n_trials, n_times)
            for band_name, (fmin, fmax) in self.bands.items():
                # Apply bandpass filter and compute power
                power_list = []
                for trial in hit_data:
                    filtered = mne.filter.filter_data(
                        trial, self.sfreq, fmin, fmax, method='iir', verbose=False
                    )
                    power = np.mean(filtered ** 2)
                    power_list.append(power)
                hit_band_powers[band_name].append(np.mean(power_list))
            
            # Miss trials
            miss_data = miss_epochs.get_data()[:, ch_idx, :]
            for band_name, (fmin, fmax) in self.bands.items():
                power_list = []
                for trial in miss_data:
                    filtered = mne.filter.filter_data(
                        trial, self.sfreq, fmin, fmax, method='iir', verbose=False
                    )
                    power = np.mean(filtered ** 2)
                    power_list.append(power)
                miss_band_powers[band_name].append(np.mean(power_list))
        
        # Average across channels
        hit_powers = {band: np.mean(powers) for band, powers in hit_band_powers.items()}
        miss_powers = {band: np.mean(powers) for band, powers in miss_band_powers.items()}
        hit_std = {band: np.std(powers) / np.sqrt(len(powers)) 
                  for band, powers in hit_band_powers.items()}
        miss_std = {band: np.std(powers) / np.sqrt(len(powers)) 
                   for band, powers in miss_band_powers.items()}
        
        # Statistical tests (use scipy.stats.ttest_ind — not in scipy.signal)
        p_values = {}
        for band in self.bands.keys():
            # Use Welch's t-test (equal_var=False) as a safer default
            _, p_val = stats.ttest_ind(hit_band_powers[band], miss_band_powers[band], equal_var=False)
            p_values[band] = float(p_val)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bands_to_plot = ['theta', 'alpha', 'beta']  # Like the example
        x = np.arange(len(bands_to_plot))
        width = 0.35
        
        hit_vals = [hit_powers[b] for b in bands_to_plot]
        miss_vals = [miss_powers[b] for b in bands_to_plot]
        hit_errs = [hit_std[b] for b in bands_to_plot]
        miss_errs = [miss_std[b] for b in bands_to_plot]
        
        bars1 = ax.bar(x - width/2, hit_vals, width, yerr=hit_errs, 
                      label='Hit (ext_1)', color='#2ecc71', alpha=0.8, 
                      capsize=5, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, miss_vals, width, yerr=miss_errs, 
                      label='Miss (ext_0)', color='#e74c3c', alpha=0.8, 
                      capsize=5, edgecolor='black', linewidth=1.5)
        
        # Add significance stars
        for i, band in enumerate(bands_to_plot):
            p_val = p_values[band]
            y_max = max(hit_vals[i] + hit_errs[i], miss_vals[i] + miss_errs[i])
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            ax.text(i, y_max * 1.12, f'p={p_val:.3f}\n{sig_text}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Band Power (μV²)', fontsize=12, fontweight='bold')
        ax.set_title(f'{session_name} — Theta / Alpha / Beta Power by Condition', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands_to_plot])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'band_power_comparison_{session_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved band power comparison: band_power_comparison_{session_name}.png")
        plt.close()
    
    def create_all_psd_plots(self, epochs, session_name):
        """
        Generate all PSD visualizations.
        
        Args:
            epochs (mne.Epochs): Epoched data
            session_name (str): Session identifier
        """
        print(f"\nGenerating PSD plots for {session_name}...")
        # Per-channel PSD plot disabled for performance (each session takes ~2-3 minutes)
        # self.plot_per_channel_psd(epochs, session_name)
        self.plot_grand_average_all_channels(epochs, session_name)
        self.plot_band_power_comparison(epochs, session_name)
        print(f"All PSD plots created for {session_name}!\n")
