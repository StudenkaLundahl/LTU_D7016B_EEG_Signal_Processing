# ============================================================================
# TASK 1.4: DATA EXPLORATION & UNDERSTANDING
# ============================================================================

class EEGDataExplorer:
    """
    Explores and displays information about EEG data structure.
    
    This class loads metadata, EEG signals, and event markers to provide
    a comprehensive overview of the dataset before processing.
    """
    
    def __init__(self, data_folder):
        """
        Initialize explorer with path to session folder.
        
        Args:
            data_folder (str): Path to session folder (e.g., '50-1')
        """
        self.data_folder = data_folder
        self.exg_file = None      # Will store path to EEG signal file
        self.marker_file = None   # Will store path to event marker file
        self.meta_file = None     # Will store path to metadata file
        self._find_files()
        
    def _find_files(self):
        """Find ExG (EEG), Marker, and Meta files in the session folder."""
        import os

        for file in os.listdir(self.data_folder):
            if file.endswith('_ExG.csv'):
                self.exg_file = os.path.join(self.data_folder, file)
            elif file.endswith('_Marker.csv'):
                self.marker_file = os.path.join(self.data_folder, file)
            elif file.endswith('_Meta.csv'):
                self.meta_file = os.path.join(self.data_folder, file)
    
    def explore_data(self):
        """
        Display key information about the dataset including:
        - Sampling rate and device information
        - EEG signal characteristics (duration, voltage ranges)
        - Performance summary (number of Hits vs Misses)
        """
        import pandas as pd
        import os
        
        print("="*80)
        print("EEG DATA EXPLORATION")
        print("="*80)
        
        # 1. Load and display metadata (sampling rate, device info)
        print("\n1. METADATA INFORMATION")
        print("-"*80)
        if self.meta_file:
            meta_df = pd.read_csv(self.meta_file)
            print(f"Sampling Rate: {meta_df['sr'].values[0]} Hz")
            print(f"Device: {meta_df['Device'].values[0]}")
            print(f"Channels: {meta_df['adcMask'].values[0]}")
            print(f"Units: {meta_df['ExGUnits'].values[0]}")
            self.sampling_rate = meta_df['sr'].values[0]
        
        # 2. Load and display EEG signal information
        print("\n2. EEG SIGNAL DATA (ExG)")
        print("-"*80)
        if self.exg_file:
            # Read first few rows to understand structure
            exg_df = pd.read_csv(self.exg_file, nrows=5)
            print(f"File: {os.path.basename(self.exg_file)}")
            print(f"Columns: {list(exg_df.columns)}")
            print(f"\nFirst few samples:")
            print(exg_df.head())
            
            # Get full dataset information
            exg_full = pd.read_csv(self.exg_file)
            print(f"\nTotal samples: {len(exg_full)}")
            print(f"Duration: {len(exg_full) / self.sampling_rate:.2f} seconds")
            print(f"Number of channels: {len(exg_df.columns) - 1}")  # -1 for TimeStamp
            
            # Display voltage ranges for each channel
            print(f"\nVoltage range (μV):")
            for ch in ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']:
                print(f"  {ch}: [{exg_full[ch].min():.2f}, {exg_full[ch].max():.2f}]")
        
        # 3. Load and display event marker information (Hit/Miss labels)
        print("\n3. EVENT MARKERS (Hit/Miss)")
        print("-"*80)
        if self.marker_file:
            marker_df = pd.read_csv(self.marker_file)
            print(f"File: {os.path.basename(self.marker_file)}")
            print(f"Total trials: {len(marker_df)}")
            
            # Calculate performance statistics
            print(f"\nPerformance summary:")
            hits = (marker_df['Code'] == 'ext_1').sum()
            misses = (marker_df['Code'] == 'ext_0').sum()
            print(f"  Hits (ext_1): {hits} ({hits/len(marker_df)*100:.1f}%)")
            print(f"  Misses (ext_0): {misses} ({misses/len(marker_df)*100:.1f}%)")
            
            print(f"\nFirst few events:")
            print(marker_df.head(10))
        
        print("\n" + "="*80)
        print("EXPLORATION COMPLETE")
        print("="*80 + "\n")