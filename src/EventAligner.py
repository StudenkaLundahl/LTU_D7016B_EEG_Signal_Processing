# ============================================================================
# TASK 1.6: EVENT ALIGNMENT (HIT/MISS MAPPING)
# ============================================================================

class EventAligner:
    """
    Aligns EEG data with behavioral events (Hit/Miss outcomes).
    
    This class handles the critical task of synchronizing event markers
    (when putts occurred) with the corresponding EEG data segments.
    """
    
    def __init__(self, raw, marker_file, eeg_first_timestamp):
        """
        Initialize aligner with EEG data and event information.
        
        Args:
            raw (MNE Raw): Preprocessed EEG data
            marker_file (str): Path to marker CSV with event timestamps
            eeg_first_timestamp (float): First timestamp from EEG file
            
        The eeg_first_timestamp is crucial because:
        - EEG and marker files use absolute timestamps (e.g., 4457.11 seconds)
        - MNE Raw objects start at time 0
        - We need to convert marker timestamps to sample indices in the Raw object
        """
        self.raw = raw
        self.marker_file = marker_file
        self.eeg_first_timestamp = eeg_first_timestamp
        self.events = None  # Will store MNE events array
        self.event_dict = {'Miss': 0, 'Hit': 1}  # Event ID mapping
        self._create_events()
    
    def _create_events(self):
        """
        Create MNE events array from marker file.
        
        MNE events format: [sample_index, 0, event_id]
        - sample_index: Position in EEG data where event occurred
        - 0: Previous event ID (not used, always 0)
        - event_id: Type of event (0=Miss, 1=Hit)
        
        Process:
        1. Load marker timestamps and codes
        2. Calculate time offset from start of EEG
        3. Convert time offset to sample index
        4. Create events array for MNE
        """
        import pandas as pd
        import numpy as np

        # Load event markers
        marker_df = pd.read_csv(self.marker_file)
        
        # Get sampling frequency from EEG data
        sfreq = self.raw.info['sfreq']
        
        # Clean up marker codes (remove extra whitespace)
        marker_df['Code'] = marker_df['Code'].str.strip()
        
        # Show alignment information for debugging
        print(f"EEG starts at: {self.eeg_first_timestamp:.2f} seconds")
        print(f"First marker at: {marker_df['TimeStamp'].iloc[0]:.2f} seconds")
        print(f"Time difference: {marker_df['TimeStamp'].iloc[0] - self.eeg_first_timestamp:.2f} seconds")
        
        # Convert each marker timestamp to a sample index
        events_list = []
        for idx, row in marker_df.iterrows():
            # Calculate how many seconds after EEG start this event occurred
            time_from_start = row['TimeStamp'] - self.eeg_first_timestamp
            
            # Convert seconds to sample index (samples = seconds × sampling_rate)
            sample_idx = int(time_from_start * sfreq)
            
            # Determine event type: ext_1 = Hit (1), ext_0 = Miss (0)
            event_id = 1 if row['Code'] == 'ext_1' else 0
            
            # Only include events that fall within the EEG recording
            if 0 <= sample_idx < len(self.raw.times):
                events_list.append([sample_idx, 0, event_id])
            else:
                print(f"⚠ Warning: Event {idx} at sample {sample_idx} is outside recording range")
        
        # Create final events array
        if len(events_list) == 0:
            print("\n⚠ WARNING: No valid events found!")
            print("Check timestamp alignment between EEG and Marker files.")
            self.events = np.array([]).reshape(0, 3)
        else:
            self.events = np.array(events_list)
            print(f"\nCreated {len(self.events)} events:")
            print(f"  Hits: {(self.events[:, 2] == 1).sum()}")
            print(f"  Misses: {(self.events[:, 2] == 0).sum()}")
    
    def create_epochs(self, tmin=-2.0, tmax=1.0, baseline=(-2.0, -1.0)):
        """
        Create epochs (time-locked segments) around each event.
        
        Args:
            tmin (float): Start time before event (seconds, negative)
            tmax (float): End time after event (seconds, positive)
            baseline (tuple): Time window for baseline correction
            
        Returns:
            MNE Epochs object or None if no events
            
        Epochs are segments of EEG data time-locked to events:
        - tmin=-2.0, tmax=1.0 gives a 3-second window around each putt
        - Baseline correction removes pre-event activity to highlight
          event-related changes
        - This allows averaging across trials to reveal patterns
        """
        import mne
        from mne import Epochs

        if len(self.events) == 0:
            print("\n⚠ Cannot create epochs: No valid events found!")
            return None
            
        print("\n" + "="*80)
        print("CREATING EPOCHS AROUND EVENTS")
        print("="*80)
        print(f"Time window: {tmin} to {tmax} seconds around event")
        print(f"Baseline correction: {baseline[0]} to {baseline[1]} seconds")
        
        # Create epochs using MNE
        # This extracts data segments around each event and applies baseline correction
        epochs = Epochs(self.raw, self.events, event_id=self.event_dict,
                       tmin=tmin, tmax=tmax, baseline=baseline,
                       preload=True, verbose=False)
        
        print(f"\nCreated epochs:")
        print(f"  Total: {len(epochs)}")
        print(f"  Hit trials: {len(epochs['Hit'])}")
        print(f"  Miss trials: {len(epochs['Miss'])}")
        print("="*80 + "\n")
        
        return epochs
    
    def plot_erp(self, epochs, session_name=None):
        """
        Plot Event-Related Potentials (ERPs) comparing Hit vs Miss trials.
        
        Args:
            epochs (MNE Epochs): Epoched data
            session_name (str): Name of the session for plot title

            
        ERPs are averaged brain responses to events:
        - Average EEG from all Hit trials
        - Average EEG from all Miss trials
        - Compare to find differences in brain activity
        - Differences might reveal neural predictors of performance
        """
        import matplotlib.pyplot as plt

        if epochs is None or len(epochs) == 0:
            print("⚠ Cannot create ERP plot: No valid epochs!")
            return
            
        # Create 8-panel figure (one per channel)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        ch_names = epochs.info['ch_names']
        
        # Plot each channel
        for idx, ch in enumerate(ch_names):
            # Calculate average ERP for Hit and Miss trials
            hit_data = epochs['Hit'].get_data(picks=ch).mean(axis=0).squeeze() * 1e6  # μV
            miss_data = epochs['Miss'].get_data(picks=ch).mean(axis=0).squeeze() * 1e6
            times = epochs.times
            
            # Plot both conditions
            axes[idx].plot(times, hit_data, label='Hit', color='green', linewidth=2)
            axes[idx].plot(times, miss_data, label='Miss', color='red', linewidth=2)
            axes[idx].axvline(0, color='black', linestyle='--', alpha=0.5, label='Event')
            axes[idx].set_xlabel('Time (s)')
            axes[idx].set_ylabel('Amplitude (μV)')
            axes[idx].set_title(f'Channel {ch}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        title = "Event-Related Potentials: Hit vs Miss"
        if session_name:
            title = f"{session_name} — {title}"
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('erp_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved ERP comparison plot: erp_comparison.png")
        plt.close()
