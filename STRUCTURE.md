# Execution Structure Guide

**How to Set Up Files to Run the EEG Analysis Code**

This guide explains the **execution structure** - how files must be organized on your local computer to run the code successfully. This is different from the GitHub repository structure (see README.md).

---

## ⚠️ CRITICAL: Execution Requirements

### **All Python files and data MUST be in the same folder: "EEG Data"**

The code expects this specific structure to run:

```
EEG Data/                                 # Main working directory
│
├── All Python modules (12 files)         # REQUIRED: Must be here to run
│   ├── AdvancedFeatureExtractor.py
│   ├── CNNEEGClassifier.py
│   ├── DeepLearningDataPreparator.py
│   ├── EEGClassifier.py
│   ├── EEGDataExplorer.py
│   ├── EEGPreprocessor.py
│   ├── EventAligner.py
│   ├── FeatureExtractor.py
│   ├── GrandAveragePSDPlotter.py
│   ├── RandomForestEEGClassifier.py
│   ├── ResultsAnalyzer.py
│   └── ThreeModelComparator.py
│
├── eeg_processing_Main_v9.py             # REQUIRED: Main script
│
├── 50-1/                                 # Session 1 data
│   ├── 2025-06-24-10-00_ExG.csv          # EEG signals (~600MB)
│   ├── 2025-06-24-10-00_Marker.csv       # Hit/Miss events
│   ├── test_Meta.csv                     # Sampling rate metadata
│   └── test_ORN.csv                      # Motion sensors (not used)
│
├── 50-2/                                 # Session 2 data
│   ├── 2025-06-24-10-30_ExG.csv
│   ├── 2025-06-24-10-30_Marker.csv
│   ├── 2025-06-24-10-30_Meta.csv
│   └── 2025-06-24-10-30_ORN.csv
│
├── 50-3/ through 50-10/                   # Sessions 3-10 data (same structure)
│
├── Read_me.txt                            # Data acquisition notes
│
├── final_results/                         # CREATED by code when running
│   ├── analysis_summary.txt               # Generated summary
│   ├── environmental_impact.csv
│   ├── environmental_impact.png
│   ├── model_comparison_comprehensive.png
│   ├── model_comparison_summary_table.csv
│   ├── model_comparison_summary_table.txt
│   ├── model_comparison_summary_table.xlsx
│   ├── performance_comparison.csv
│   ├── statistical_tests.json
│   ├── three_model_comparison.png
│   ├── three_model_performance.csv
│   ├── three_model_statistics.json
│   └── three_model_summary.txt
│
└── *.png files (generated in root)        # CREATED by code when running
    ├── preprocessing_comparison_501.png
    ├── preprocessing_comparison_502.png
    ├── ... (10 preprocessing plots)
    ├── erp_comparison_501.png
    ├── erp_comparison_502.png
    ├── ... (10 ERP plots)
    ├── band_power_comparison_501.png
    ├── ... (10 band power plots)
    ├── psd_grand_average_501.png
    ├── ... (10 PSD plots)
    ├── feature_importance_rf_501.png
    ├── ... (10 RF feature plots)
    ├── psd_feature_importance_501.png
    ├── ... (10 PSD feature plots)
    ├── confusion_matrix_lr_501.png
    ├── confusion_matrix_rf_501.png
    ├── confusion_matrix_cnn_501.png
    ├── ... (30 confusion matrices total)
    ├── cnn_training_history_501.png
    └── ... (10 CNN training plots)

```

---

## 🚀 How to Run the Code

### Step 1: Organize Your Files

Create a folder called **"EEG Data"** and place these items inside:

#### **Required Files to Copy:**
1. **All 12 Python module files** (from `src/` in GitHub)
   - AdvancedFeatureExtractor.py
   - CNNEEGClassifier.py
   - DeepLearningDataPreparator.py
   - EEGClassifier.py
   - EEGDataExplorer.py
   - EEGPreprocessor.py
   - EventAligner.py
   - FeatureExtractor.py
   - GrandAveragePSDPlotter.py
   - RandomForestEEGClassifier.py
   - ResultsAnalyzer.py
   - ThreeModelComparator.py

2. **Main execution script**
   - eeg_processing_Main_v9.py

3. **Your 10 session folders**
   - 50-1/ through 50-10/ (each with ExG, Marker, Meta, ORN files)

#### **Final Structure Check:**
```
Your Computer/
└── EEG Data/
    ├── All 12 .py module files  ← Python modules
    ├── eeg_processing_Main_v9.py ← Main script
    └── 50-1/ through 50-10/      ← Data folders
```

### Step 2: Run the Analysis

**Option 1: Command Line**
```bash
cd "path/to/EEG Data"
python eeg_processing_Main_v9.py
```

**Option 2: Jupyter Notebook**
```bash
cd "path/to/EEG Data"
jupyter notebook eeg_processing_Main_v9.ipynb
```

### Step 3: Outputs Generated

When code finishes running, you'll have:

**In "EEG Data/" root folder:**
- ~100 PNG plot files (10 sessions × ~10 plot types each)
- Examples: `erp_comparison_501.png`, `confusion_matrix_rf_505.png`, etc.

**In "EEG Data/final_results/" folder:**
- Summary statistics (TXT, CSV, JSON, XLSX files)
- Key comparison plots (three_model_comparison.png, environmental_impact.png, model_comparison_comprehensive.png)

---

## 📁 Understanding the File Organization

### Why This Structure?

The code uses **relative imports** and expects files in specific locations:

```python
# In eeg_processing_Main_v9.py
from EEGPreprocessor import EEGPreprocessor           # Looks in same folder
from AdvancedFeatureExtractor import ...              # Looks in same folder
...

# Data loading
session_folders = ['50-1', '50-2', ..., '50-10']     # Looks in same folder
```

**If Python modules are not in "EEG Data" folder → Import errors!**

### File Categories

| Category | Location | Created By | Size |
|----------|----------|------------|------|
| **Python modules** | EEG Data/ root | You copy these | ~50KB total |
| **Main script** | EEG Data/ root | You copy this | ~20KB |
| **Data folders** | EEG Data/50-X/ | Your experiment | ~600MB per session |
| **Plot outputs** | EEG Data/*.png | Code generates | ~500KB per plot |
| **Summary outputs** | EEG Data/final_results/ | Code generates | ~2MB total |

---

## 📊 Data File Details

### Files in Each Session Folder (50-1 through 50-10)

```
50-X/
├── YYYY-MM-DD-HH-MM_ExG.csv      # 8-channel EEG recordings
│   ├── Columns: TimeStamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8
│   ├── Size: ~600MB
│   ├── Sampling rate: 500 Hz
│   └── Used by: EEGPreprocessor.py ✅
│
├── YYYY-MM-DD-HH-MM_Marker.csv   # Event timestamps and labels
│   ├── Columns: TimeStamp, Code
│   ├── Codes: ext_0 (Miss), ext_1 (Hit)
│   ├── Size: ~1KB
│   └── Used by: EventAligner.py ✅
│
├── YYYY-MM-DD-HH-MM_Meta.csv     # Recording metadata
│   ├── Columns: TimeOffset, Device, sr, adcMask, ExGUnits
│   ├── Size: ~1KB
│   └── Used by: EEGPreprocessor.py (to get sampling rate 500 Hz) ✅
│
└── YYYY-MM-DD-HH-MM_ORN.csv      # Orientation sensor data
    ├── Columns: TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz
    ├── Size: ~50MB
    └── Used by: NONE ❌ (not used in current analysis)
```

### Session-Specific Notes

**Session 50-5** (optimal quality):
- Has additional file: `Impedance.txt` (documents impedance: 12-20 kΩ)

**Session 50-8** (sunlight interference):
- Has additional file: `Note.txt` (documents environmental issue)

**Sessions 50-9, 50-10** (participant hunger):
- Has additional file: `Note.txt` (documents participant state)

---

## 🔧 Module Dependencies

### Module Import Chain

```
eeg_processing_Main_v9.py
  ├─→ EEGPreprocessor.py
  ├─→ EventAligner.py
  ├─→ FeatureExtractor.py (Grade 3)
  ├─→ AdvancedFeatureExtractor.py (Grade 4)
  ├─→ EEGClassifier.py (Grade 3)
  ├─→ RandomForestEEGClassifier.py (Grade 4)
  ├─→ DeepLearningDataPreparator.py (Grade 5)
  ├─→ CNNEEGClassifier.py (Grade 5)
  ├─→ GrandAveragePSDPlotter.py
  ├─→ ResultsAnalyzer.py
  └─→ ThreeModelComparator.py
```

**All modules must be in same directory for imports to work!**

---

## 📈 Output Files Generated

### PNG Plots in Root (Total: ~100 files)

**Per-Session Plots** (10 copies each):
```
preprocessing_comparison_50X.png       # Raw vs filtered EEG signals
erp_comparison_50X.png                 # Event-Related Potentials (Hit vs Miss)
band_power_comparison_50X.png          # Theta/Alpha/Beta band powers
psd_grand_average_50X.png              # Power Spectral Density across channels
feature_importance_rf_50X.png          # Top 30 Random Forest features
psd_feature_importance_50X.png         # PSD feature distribution
confusion_matrix_lr_50X.png            # Logistic Regression performance
confusion_matrix_rf_50X.png            # Random Forest performance
confusion_matrix_cnn_50X.png           # CNN performance
cnn_training_history_50X.png           # Training/validation curves
```

Where X = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (session number)

### Summary Files in final_results/ Folder

**Statistical Summaries:**
```
three_model_summary.txt                # Human-readable statistical tests
three_model_statistics.json            # Detailed metrics (parseable)
three_model_performance.csv            # Session-by-session results
statistical_tests.json                 # Paired t-tests, Cohen's d
analysis_summary.txt                   # Overall analysis summary
```

**Performance Tables:**
```
model_comparison_summary_table.csv     # Performance comparison (CSV)
model_comparison_summary_table.txt     # Performance comparison (text)
model_comparison_summary_table.xlsx    # Performance comparison (Excel)
performance_comparison.csv             # LR vs RF detailed comparison
environmental_impact.csv               # Quality factors per session
```

**Key Plots:**
```
three_model_comparison.png             # 6-panel comprehensive comparison LR vs RF vs CNN
model_comparison_comprehensive.png     # Alternative visualization LR vs RF
environmental_impact.png               # Quality vs performance correlation
```

---

## 🔄 Typical Workflow

### 1. Initial Setup
```bash
# Create working directory
mkdir "EEG Data"
cd "EEG Data"

# Copy all Python files here
# Copy all session folders (50-1 through 50-10) here
```

### 2. Run Analysis
```bash
python eeg_processing_Main_v9.py
```

### 3. Check Outputs
```bash
# View plots
ls *.png

# View summaries
cd final_results
ls

```

### 4. Results Review
- Open plots in image viewer
- Open Excel tables in Microsoft Excel
- Read summary text files
- Analyze statistical JSON files

---

## 📝 Summary

### Key Points:
1. ✅ **All Python files** must be in "EEG Data" folder
2. ✅ **Main script** must be in "EEG Data" folder
3. ✅ **Session folders** (50-1 through 50-10) must be in "EEG Data" folder
4. ✅ **Run from** "EEG Data" folder: `python eeg_processing_Main_v9.py`
5. ✅ **Outputs appear** in "EEG Data" root (plots) and "EEG Data/final_results/" (summaries)

### This is NOT GitHub Structure:
- GitHub repository has `src/`, `notebooks/`, `docs/` folders (organized for viewing)
- Execution structure has everything in "EEG Data" folder (organized for running)
- When sharing code on GitHub, use clean structure from README.md
- When running code locally, use this execution structure

---

## 🎯 Quick Reference

```bash
# Your execution setup:
EEG Data/
├── AdvancedFeatureExtractor.py      # Copy from GitHub src/
├── CNNEEGClassifier.py              # Copy from GitHub src/
├── ... (all other modules)          # Copy from GitHub src/
├── eeg_processing_Main_v9.py        # Copy from GitHub root
├── 50-1/ through 50-10/             # Your data folders
└── final_results/                   # Created by code
    └── (summary files)

# To run:
cd "EEG Data"
python eeg_processing_Main_v9.py

# Results appear in:
EEG Data/*.png                       # Plots in root
EEG Data/final_results/              # Summaries in subfolder
```

---

*This structure guide ensures successful code execution. For GitHub organization, see README.md.*

**Course**: D7016B Signal Processing and Machine Learning  
**Student**: Studenka Lundahl  
**Institution**: Luleå University of Technology  
**December 2025**
