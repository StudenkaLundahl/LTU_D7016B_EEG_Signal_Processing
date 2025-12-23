# ============================================================================
# TASK 1.5: BASIC DATA PREPROCESSING (FILTERING, ARTIFACTS)
# ============================================================================

class EEGPreprocessor:
    """
    Preprocesses EEG data using MNE library.
    
    Applies standard EEG preprocessing steps:
    - Notch filtering to remove power line noise (50 Hz)
    - Bandpass filtering to keep relevant EEG frequencies (1-40 Hz)
    """
    
    def __init__(self, exg_file, meta_file):
        """
        Initialize preprocessor with EEG and metadata files.
        
        Args:
            exg_file (str): Path to EEG signal CSV file
            meta_file (str): Path to metadata CSV file
        """
        self.exg_file = exg_file
        self.meta_file = meta_file
        self.raw = None              # Will store original MNE Raw object
        self.raw_filtered = None     # Will store filtered MNE Raw object
        self.first_timestamp = None  # Store first EEG timestamp for event alignment
        self._load_data()
    
    def _load_data(self):
        """
        Load EEG data and create MNE Raw object.
        
        Steps:
        1. Read metadata to get sampling rate
        2. Load EEG channel data
        3. Convert voltages from μV to V (MNE standard)
        4. Create MNE Raw object with proper channel info
        5. Set standard 10-20 electrode positions
        """
        import pandas as pd
        import mne
        from mne.io import RawArray
        from mne import create_info, Epochs

        # Load sampling rate from metadata
        meta_df = pd.read_csv(self.meta_file)
        sfreq = meta_df['sr'].values[0]
        
        # Load EEG signal data
        exg_df = pd.read_csv(self.exg_file)
        
        # Store first timestamp - needed for aligning events with EEG data
        self.first_timestamp = exg_df['TimeStamp'].iloc[0]
        
        # Extract channel data (ch1-ch8) and convert to Volts
        # MNE expects data in Volts, but our data is in microvolts
        ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        data = exg_df[['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']].values.T
        data = data * 1e-6  # Convert from μV to V
        
        # Create MNE info structure with channel names and sampling rate
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        
        # Create MNE Raw object (main data structure for continuous EEG)
        self.raw = RawArray(data, info)
        
        # Set standard 10-20 montage for electrode positions
        # This allows MNE to know where electrodes are on the scalp
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)
        
        print(f"Loaded EEG data: {len(ch_names)} channels, {sfreq} Hz")
        print(f"EEG recording starts at timestamp: {self.first_timestamp:.2f} seconds")
    
    def preprocess(self, l_freq=1.0, h_freq=40.0, notch_freq=50.0):
        """
        Apply preprocessing filters to remove noise and artifacts.
        
        Args:
            l_freq (float): High-pass filter cutoff (Hz) - removes slow drifts
            h_freq (float): Low-pass filter cutoff (Hz) - removes high-freq noise
            notch_freq (float): Notch filter frequency (Hz) - removes power line noise
            
        Returns:
            MNE Raw object: Filtered EEG data
            
        Filter explanations:
        - Notch filter (50 Hz): Removes electrical interference from power lines
        - Bandpass (1-40 Hz): Keeps relevant brain activity frequencies
          * < 1 Hz: Slow drifts and DC offset (artifacts)
          * 1-40 Hz: Brain activity (delta, theta, alpha, beta waves)
          * > 40 Hz: Muscle artifacts and high-frequency noise
        """
        print("\n" + "="*80)
        print("PREPROCESSING EEG DATA")
        print("="*80)
        
        # Make a copy for filtering (preserves original data)
        self.raw_filtered = self.raw.copy()
        
        # Step 1: Remove power line noise at 50 Hz (60 Hz in North America)
        print(f"\n1. Applying notch filter at {notch_freq} Hz (power line noise)...")
        self.raw_filtered.notch_filter(freqs=notch_freq, verbose=False)
        
        # Step 2: Apply bandpass filter to keep only relevant brain frequencies
        print(f"2. Applying bandpass filter: {l_freq}-{h_freq} Hz...")
        self.raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        
        print("\nPreprocessing complete!")
        print("="*80 + "\n")
        
        return self.raw_filtered
    
    def plot_comparison(self, duration=10.0, start=0.0, session_name=None):
        """
        Create visualization comparing raw vs filtered data.
        
        Args:
            duration (float): Duration of data to plot (seconds)
            start (float): Start time for plotting (seconds)
            session_name (str): Name of the session for plot title
            
        This plot helps visualize the effect of preprocessing by showing
        the same data segment before and after filtering.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Get data segment for plotting
        times = self.raw.times
        start_idx = np.searchsorted(times, start)
        end_idx = np.searchsorted(times, start + duration)
        
        # Extract channel names and data
        ch_names = self.raw.info['ch_names']
        raw_data = self.raw.get_data()[:, start_idx:end_idx] * 1e6  # Convert to μV
        filtered_data = self.raw_filtered.get_data()[:, start_idx:end_idx] * 1e6
        plot_times = times[start_idx:end_idx]
        
        # Create figure with two subplots (raw and filtered)
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot raw data (top panel)
        # Each channel is offset vertically for visibility
        for i, ch_name in enumerate(ch_names):
            axes[0].plot(plot_times, raw_data[i, :] + i * 100, label=ch_name, linewidth=0.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (μV) + offset')
        axes[0].set_title('Raw EEG Data', fontsize=14, fontweight='bold')
        axes[0].legend(loc='right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot filtered data (bottom panel)
        for i, ch_name in enumerate(ch_names):
            axes[1].plot(plot_times, filtered_data[i, :] + i * 100, label=ch_name, linewidth=0.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (μV) + offset')
        axes[1].set_title('Filtered EEG Data (1-40 Hz + Notch 50Hz)', fontsize=14, fontweight='bold')
        axes[1].legend(loc='right')
        axes[1].grid(True, alpha=0.3)
        
        if session_name:
            fig.suptitle(f"{session_name} — Preprocessing Comparison", fontsize=14, fontweight="bold")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved preprocessing comparison plot: preprocessing_comparison.png")
        plt.close()
