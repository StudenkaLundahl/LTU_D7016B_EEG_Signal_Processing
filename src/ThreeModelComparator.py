"""
Three-Model Comparator Module
Comprehensive comparison of Logistic Regression, Random Forest, and CNN

Student: Studenka Lundahl
Supervisor: Adoul Mohammed Amin
Date: December 15, 2025

Description:
This module provides comprehensive statistical and visual comparison of three
classification models (Logistic Regression, Random Forest, and CNN) across
multiple EEG sessions. It generates performance tables, statistical tests,
and publication-quality comparison plots.

Key Features:
- Session-by-session performance tracking for all three models
- Paired statistical tests (t-test, Cohen's d effect size)
- ANOVA for overall model comparison
- Comprehensive visualization with side-by-side comparisons
- Summary tables in multiple formats (Excel, CSV, TXT)
- Robust handling of missing CNN data (graceful degradation)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


class ThreeModelComparator:
    """
    Compare three classification models: Logistic Regression, Random Forest, and CNN.
    
    This class aggregates results from multiple sessions and provides statistical
    analysis and visualization comparing the three different modeling approaches.
    It handles cases where CNN results may be unavailable for some sessions.
    """
    
    def __init__(self, output_dir='final_results'):
        """
        Initialize the three-model comparator.
        
        Args:
            output_dir (str): Directory to save comparison plots and results
        """
        self.output_dir = output_dir
        self.session_results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"ThreeModelComparator initialized")

    def add_session_results(self, session_name, lr_metrics, rf_metrics, cnn_metrics=None):
        """
        Add a single session's results to the comparator.
        
        This method collects metrics from all three models for each session.
        CNN metrics are optional to handle cases where CNN training may not be complete.
        
        Args:
            session_name (str): Session identifier (e.g., '50-1')
            lr_metrics (dict): Logistic Regression metrics from evaluate_model()
            rf_metrics (dict): Random Forest metrics from evaluate_model()
            cnn_metrics (dict, optional): CNN metrics from evaluate_model()
        
        The metrics dictionaries should contain:
            - 'test_accuracy': Test set accuracy
            - 'cv_mean': Cross-validation mean accuracy
            - 'f1_score': F1 score for positive class
        """
        entry = {
            'session': session_name,
            'lr_test_acc': float(lr_metrics.get('test_accuracy')) if lr_metrics and lr_metrics.get('test_accuracy') is not None else np.nan,
            'rf_test_acc': float(rf_metrics.get('test_accuracy')) if rf_metrics and rf_metrics.get('test_accuracy') is not None else np.nan,
            'lr_cv': float(lr_metrics.get('cv_mean')) if lr_metrics and lr_metrics.get('cv_mean') is not None else np.nan,
            'rf_cv': float(rf_metrics.get('cv_mean')) if rf_metrics and rf_metrics.get('cv_mean') is not None else np.nan,
            'lr_f1': float(lr_metrics.get('f1_score')) if lr_metrics and lr_metrics.get('f1_score') is not None else np.nan,
            'rf_f1': float(rf_metrics.get('f1_score')) if rf_metrics and rf_metrics.get('f1_score') is not None else np.nan,
        }

        # Add CNN metrics if available
        if cnn_metrics:
            entry['cnn_test_acc'] = float(cnn_metrics.get('test_accuracy')) if cnn_metrics.get('test_accuracy') is not None else np.nan
            entry['cnn_cv'] = float(cnn_metrics.get('cv_mean')) if cnn_metrics.get('cv_mean') is not None else np.nan
            entry['cnn_f1'] = float(cnn_metrics.get('f1_score')) if cnn_metrics.get('f1_score') is not None else np.nan
        else:
            entry['cnn_test_acc'] = np.nan
            entry['cnn_cv'] = np.nan
            entry['cnn_f1'] = np.nan

        self.session_results.append(entry)
        print(f"Added session results: {session_name}")

    def generate_comprehensive_comparison(self):
        """
        Generate complete three-model comparison including plots, tables, and statistics.
        
        This is the main entry point that produces:
        1. Comprehensive comparison plot (test accuracy, F1 scores, cross-validation)
        2. Summary tables in multiple formats (Excel, CSV, TXT)
        3. Statistical significance tests (paired t-tests, ANOVA, effect sizes)
        4. Text summary with interpretation
        
        The method is designed to be robust - plotting errors won't prevent
        saving tables and statistical results.
        """
        if not self.session_results:
            print("No session results to compare.")
            return

        # Convert results to DataFrame for analysis
        df = pd.DataFrame(self.session_results)
        total_sessions = len(df)
        has_cnn = df['cnn_test_acc'].notna().any()

        # Calculate descriptive stats
        stats_summary = self._calculate_statistics(df, has_cnn)

        # Perform statistical tests (guarded)
        try:
            statistical_tests = self._perform_statistical_tests(df, has_cnn)
        except Exception as e:
            print(f"⚠️ Warning: Statistical tests failed: {e}")
            statistical_tests = {}

        # Create visualization (guarded)
        try:
            self._create_comprehensive_plot(df, has_cnn, stats_summary)
        except Exception as e:
            print(f"⚠️ Warning: Could not create comprehensive plot: {e}")

        # Save detailed CSV/JSON
        try:
            self._save_detailed_results(df, stats_summary, statistical_tests, has_cnn)
        except Exception as e:
            print(f"⚠️ Warning: Could not save detailed results: {e}")

        # Create and save summary table (Excel/CSV/TXT)
        try:
            self._create_summary_table(df, has_cnn)
        except Exception as e:
            print(f"⚠️ Warning: Could not create summary table: {e}")

        # Print summary to console and save
        try:
            self._print_summary(stats_summary, statistical_tests, has_cnn, total_sessions)
        except Exception as e:
            print(f"⚠️ Warning: Could not print/save summary: {e}")

    def _calculate_statistics(self, df, has_cnn):
        """
        Calculate summary statistics for each model.
        
        Computes mean accuracy, standard deviation, mean F1 score, and
        mean cross-validation accuracy for each model across all sessions.
        
        Args:
            df (pd.DataFrame): Session results dataframe
            has_cnn (bool): Whether CNN results are available
            
        Returns:
            dict: Dictionary with statistics for 'lr', 'rf', and optionally 'cnn'
        """
        stats_summary = {}

        # Logistic Regression statistics
        stats_summary['lr'] = {
            'mean_acc': float(df['lr_test_acc'].mean()),
            'std_acc': float(df['lr_test_acc'].std()),
            'mean_f1': float(df['lr_f1'].mean()),
            'mean_cv': float(df['lr_cv'].mean()) if 'lr_cv' in df.columns else float(np.nan)
        }

        # Random Forest statistics
        stats_summary['rf'] = {
            'mean_acc': float(df['rf_test_acc'].mean()),
            'std_acc': float(df['rf_test_acc'].std()),
            'mean_f1': float(df['rf_f1'].mean()),
            'mean_cv': float(df['rf_cv'].mean()) if 'rf_cv' in df.columns else float(np.nan)
        }

        # CNN statistics (only for sessions where CNN results exist)
        if has_cnn:
            cnn_data = df[df['cnn_test_acc'].notna()]
            stats_summary['cnn'] = {
                'mean_acc': float(cnn_data['cnn_test_acc'].mean()) if len(cnn_data) > 0 else float(np.nan),
                'std_acc': float(cnn_data['cnn_test_acc'].std()) if len(cnn_data) > 0 else float(np.nan),
                'mean_f1': float(cnn_data['cnn_f1'].mean()) if len(cnn_data) > 0 else float(np.nan),
                'mean_cv': float(cnn_data['cnn_cv'].mean()) if (len(cnn_data) > 0 and 'cnn_cv' in cnn_data.columns) else float(np.nan),
                'n_sessions': int(len(cnn_data))
            }

        return stats_summary

    def _perform_statistical_tests(self, df, has_cnn):
        """
        Perform paired statistical tests between models.
        
        Conducts:
        - Paired t-tests between each pair of models
        - Cohen's d effect size calculations
        - ANOVA across all three models (if all sessions have CNN results)
        
        Args:
            df (pd.DataFrame): Session results dataframe
            has_cnn (bool): Whether CNN results are available
            
        Returns:
            dict: Dictionary containing test results for each comparison
        """
        tests = {}

        # Logistic Regression vs Random Forest (paired t-test)
        # Uses all sessions since both models are always trained
        t_stat_lr_rf, p_val_lr_rf = stats.ttest_rel(df['lr_test_acc'], df['rf_test_acc'])
        cohens_d_lr_rf = self._cohens_d(df['lr_test_acc'], df['rf_test_acc'])

        tests['lr_vs_rf'] = {
            't_statistic': float(t_stat_lr_rf),
            'p_value': float(p_val_lr_rf),
            'cohens_d': float(cohens_d_lr_rf),
            'significant': bool(p_val_lr_rf < 0.05)
        }

        if has_cnn:
            # Get only sessions where CNN results are available for fair comparison
            cnn_data = df[df['cnn_test_acc'].notna()]

            # Logistic Regression vs CNN (paired t-test)
            t_stat_lr_cnn, p_val_lr_cnn = stats.ttest_rel(
                df.loc[cnn_data.index, 'lr_test_acc'],
                cnn_data['cnn_test_acc']
            )
            cohens_d_lr_cnn = self._cohens_d(
                df.loc[cnn_data.index, 'lr_test_acc'],
                cnn_data['cnn_test_acc']
            )

            tests['lr_vs_cnn'] = {
                't_statistic': float(t_stat_lr_cnn),
                'p_value': float(p_val_lr_cnn),
                'cohens_d': float(cohens_d_lr_cnn),
                'significant': bool(p_val_lr_cnn < 0.05)
            }

            # Random Forest vs CNN (paired t-test)
            t_stat_rf_cnn, p_val_rf_cnn = stats.ttest_rel(
                df.loc[cnn_data.index, 'rf_test_acc'],
                cnn_data['cnn_test_acc']
            )
            cohens_d_rf_cnn = self._cohens_d(
                df.loc[cnn_data.index, 'rf_test_acc'],
                cnn_data['cnn_test_acc']
            )

            tests['rf_vs_cnn'] = {
                't_statistic': float(t_stat_rf_cnn),
                'p_value': float(p_val_rf_cnn),
                'cohens_d': float(cohens_d_rf_cnn),
                'significant': bool(p_val_rf_cnn < 0.05)
            }

            # ANOVA (Analysis Of VAriance) across all three models (only if all sessions have CNN)
            if len(cnn_data) == len(df):
                f_stat, p_val_anova = stats.f_oneway(
                    df['lr_test_acc'],
                    df['rf_test_acc'],
                    df['cnn_test_acc']
                )
                tests['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_val_anova),
                    'significant': bool(p_val_anova < 0.05)
                }

        return tests

    def _cohens_d(self, group1, group2):
        """
        Calculate Cohen's d effect size for paired samples.
        
        For paired data (same sessions, different models), Cohen's d is calculated as:
        d = mean(differences) / std(differences)
        
        Effect size interpretation:
        - Small: |d| = 0.2
        - Medium: |d| = 0.5
        - Large: |d| = 0.8
        """
        diff = group1 - group2
        return np.mean(diff) / np.std(diff, ddof=1)
    
    def _create_comprehensive_plot(self, df, has_cnn, stats_summary):
        """
        Create comprehensive comparison visualization with 8 subplots.
        
        Displays test accuracy, cross-validation, F1 scores, and statistical
        summaries for all three models across sessions.
        """
        
        # Determine number of models
        n_models = 3 if has_cnn else 2
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Session-by-session accuracy comparison (top left, span 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        x = np.arange(len(df))
        width = 0.25 if has_cnn else 0.35
        
        ax1.bar(x - width, df['lr_test_acc'], width, label='Logistic Regression', 
                color='#3498db', alpha=0.8)
        ax1.bar(x, df['rf_test_acc'], width, label='Random Forest', 
                color='#e74c3c', alpha=0.8)
        if has_cnn:
            # Handle missing CNN values gracefully
            cnn_values = df['cnn_test_acc'].fillna(0)
            ax1.bar(x + width, cnn_values, width, label='CNN', 
                    color='#2ecc71', alpha=0.8)
        
        ax1.set_xlabel('Session', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Model Comparison Across Sessions', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['session'], rotation=45)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.0])
        
        # Subplot 2: Mean accuracy with error bars (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        models = ['LR', 'RF']
        means = [stats_summary['lr']['mean_acc'], stats_summary['rf']['mean_acc']]
        stds = [stats_summary['lr']['std_acc'], stats_summary['rf']['std_acc']]
        colors = ['#3498db', '#e74c3c']
        
        if has_cnn:
            models.append('CNN')
            means.append(stats_summary['cnn']['mean_acc'])
            stds.append(stats_summary['cnn']['std_acc'])
            colors.append('#2ecc71')
        
        ax2.bar(models, means, color=colors, alpha=0.8, yerr=stds, capsize=10)
        ax2.set_ylabel('Mean Test Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Performance ± SD', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.0])
        
        # Add values on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(i, mean + std + 0.02, f'{mean:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Subplot 3: F1-Score comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.bar(x - width, df['lr_f1'], width, label='LR', color='#3498db', alpha=0.8)
        ax3.bar(x, df['rf_f1'], width, label='RF', color='#e74c3c', alpha=0.8)
        if has_cnn:
            # Show CNN bars only where data exists
            cnn_f1_values = df['cnn_f1'].fillna(0)
            ax3.bar(x + width, cnn_f1_values, width, label='CNN', color='#2ecc71', alpha=0.8)
        
        ax3.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax3.set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(df['session'], rotation=45, fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1.0])
        
        # Subplot 4: Cross-validation scores (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.bar(x - width, df['lr_cv'], width, label='LR', color='#3498db', alpha=0.8)
        ax4.bar(x, df['rf_cv'], width, label='RF', color='#e74c3c', alpha=0.8)
        if has_cnn:
            cv_values = df['cnn_cv'].fillna(0)
            ax4.bar(x + width, cv_values, width, label='CNN', color='#2ecc71', alpha=0.8)
        
        ax4.set_ylabel('CV Accuracy', fontsize=11, fontweight='bold')
        ax4.set_title('Cross-Validation Performance', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(df['session'], rotation=45, fontsize=9)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Subplot 5: Accuracy scatter (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.scatter(df['lr_test_acc'], df['rf_test_acc'], s=100, alpha=0.6, 
                   color='purple', edgecolors='black', linewidth=1.5, label='LR vs RF')
        if has_cnn:
            cnn_mask = df['cnn_test_acc'].notna()
            ax5.scatter(df.loc[cnn_mask, 'lr_test_acc'], df.loc[cnn_mask, 'cnn_test_acc'], 
                       s=100, alpha=0.6, color='orange', edgecolors='black', linewidth=1.5, 
                       label='LR vs CNN', marker='^')
        
        # Diagonal line (equal performance)
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
        ax5.set_xlabel('Logistic Regression Accuracy', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Other Model Accuracy', fontsize=11, fontweight='bold')
        ax5.set_title('Accuracy Scatter Plot', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])
        
        # Subplot 6: Winner distribution (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        winners = []
        for _, row in df.iterrows():
            accs = [row['lr_test_acc'], row['rf_test_acc']]
            labels = ['LR', 'RF']
            if has_cnn and pd.notna(row['cnn_test_acc']):
                accs.append(row['cnn_test_acc'])
                labels.append('CNN')
            
            max_acc = max(accs)
            winner_idx = accs.index(max_acc)
            winners.append(labels[winner_idx])
        
        winner_counts = pd.Series(winners).value_counts()
        colors_map = {'LR': '#3498db', 'RF': '#e74c3c', 'CNN': '#2ecc71'}
        colors_winner = [colors_map[w] for w in winner_counts.index]
        
        ax6.bar(winner_counts.index, winner_counts.values, color=colors_winner, alpha=0.8)
        ax6.set_ylabel('Number of Sessions Won', fontsize=11, fontweight='bold')
        ax6.set_title('Winner Distribution', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, count in enumerate(winner_counts.values):
            ax6.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Subplot 7: Accuracy difference (bottom center) - RF vs LR only
        ax7 = fig.add_subplot(gs[2, 1])
        diff_rf_lr = df['rf_test_acc'] - df['lr_test_acc']
        colors_diff = ['green' if d > 0 else 'red' for d in diff_rf_lr]
        ax7.bar(x, diff_rf_lr, color=colors_diff, alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax7.set_xlabel('Session', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Accuracy Difference (RF - LR)', fontsize=11, fontweight='bold')
        ax7.set_title('RF vs LR Performance Delta', fontsize=11, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(df['session'], rotation=45, fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Subplot 8: Statistical summary (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        total_sessions = len(df)
        summary_text = "STATISTICAL SUMMARY\n" + "="*30 + "\n\n"
        summary_text += f"Total Sessions: {total_sessions}\n\n"
        summary_text += f"Logistic Regression:\n"
        summary_text += f"  Mean Acc: {stats_summary['lr']['mean_acc']:.3f} ± {stats_summary['lr']['std_acc']:.3f}\n"
        summary_text += f"  Mean F1:  {stats_summary['lr']['mean_f1']:.3f}\n\n"
        
        summary_text += f"Random Forest:\n"
        summary_text += f"  Mean Acc: {stats_summary['rf']['mean_acc']:.3f} ± {stats_summary['rf']['std_acc']:.3f}\n"
        summary_text += f"  Mean F1:  {stats_summary['rf']['mean_f1']:.3f}\n\n"
        
        if has_cnn:
            summary_text += f"CNN:\n"
            summary_text += f"  Mean Acc: {stats_summary['cnn']['mean_acc']:.3f} ± {stats_summary['cnn']['std_acc']:.3f}\n"
            summary_text += f"  Mean F1:  {stats_summary['cnn']['mean_f1']:.3f}\n\n"
        
        # Find best model
        best_model = 'LR'
        best_acc = stats_summary['lr']['mean_acc']
        if stats_summary['rf']['mean_acc'] > best_acc:
            best_model = 'RF'
            best_acc = stats_summary['rf']['mean_acc']
        if has_cnn and stats_summary['cnn']['mean_acc'] > best_acc:
            best_model = 'CNN'
            best_acc = stats_summary['cnn']['mean_acc']
        
        summary_text += f"Best Model: {best_model}\n"
        summary_text += f"  Accuracy: {best_acc:.3f}"
        
        # Position at top of subplot with vertical alignment
        ax8.text(0.1, 0.95, summary_text, fontsize=10, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Overall title
        fig.suptitle('Comprehensive Three-Model Comparison: LR vs RF vs CNN', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'three_model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comprehensive visualization saved: {output_path}")
    
    def _create_summary_table(self, df, has_cnn):
        """
        Create comprehensive summary table in Excel, CSV, and TXT formats.
        
        Generates a detailed comparison table showing test accuracy, cross-validation
        accuracy, and F1 scores for all models across all sessions.
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE SUMMARY TABLE")
        print("="*80 + "\n")
        
        # Prepare data for table
        table_data = []
        
        for _, row in df.iterrows():
            session_data = {
                'Session': row['session'],
                'LR_Test_%': f"{row['lr_test_acc']*100:.1f}",
                'RF_Test_%': f"{row['rf_test_acc']*100:.1f}",
                'LR_CV_%': f"{row['lr_cv']*100:.1f}",
                'RF_CV_%': f"{row['rf_cv']*100:.1f}",
                'LR_F1': f"{row['lr_f1']:.3f}",
                'RF_F1': f"{row['rf_f1']:.3f}",
            }
            
            # Add CNN if available
            if has_cnn:
                if pd.notna(row['cnn_test_acc']):
                    session_data['CNN_Test_%'] = f"{row['cnn_test_acc']*100:.1f}"
                    session_data['CNN_F1'] = f"{row['cnn_f1']:.3f}" if pd.notna(row['cnn_f1']) else "N/A"
                    session_data['CNN_CV_%'] = f"{row['cnn_cv']*100:.1f}" if pd.notna(row['cnn_cv']) else "N/A"
                else:
                    session_data['CNN_Test_%'] = "N/A"
                    session_data['CNN_F1'] = "N/A"
                    session_data['CNN_CV_%'] = "N/A"
            
            # Determine winner
            accs = [row['lr_test_acc'], row['rf_test_acc']]
            labels = ['LR', 'RF']
            if has_cnn and pd.notna(row['cnn_test_acc']):
                accs.append(row['cnn_test_acc'])
                labels.append('CNN')
            
            max_acc = max(accs)
            winner_idx = accs.index(max_acc)
            session_data['Winner'] = labels[winner_idx]
            
            table_data.append(session_data)
        
        # Add MEAN row
        mean_row = {
            'Session': 'MEAN',
            'LR_Test_%': f"{df['lr_test_acc'].mean()*100:.1f}",
            'RF_Test_%': f"{df['rf_test_acc'].mean()*100:.1f}",
            'LR_CV_%': f"{df['lr_cv'].mean()*100:.1f}",
            'RF_CV_%': f"{df['rf_cv'].mean()*100:.1f}",
            'LR_F1': f"{df['lr_f1'].mean():.3f}",
            'RF_F1': f"{df['rf_f1'].mean():.3f}",
        }
        
        if has_cnn:
            cnn_data = df[df['cnn_test_acc'].notna()]
            mean_row['CNN_Test_%'] = f"{cnn_data['cnn_test_acc'].mean()*100:.1f}" if len(cnn_data) > 0 else "N/A"
            mean_row['CNN_CV_%'] = f"{cnn_data['cnn_cv'].mean()*100:.1f}" if (len(cnn_data) > 0 and cnn_data['cnn_cv'].notna().any()) else "N/A"
            mean_row['CNN_F1'] = f"{cnn_data['cnn_f1'].mean():.3f}" if len(cnn_data) > 0 else "N/A"
        
        # Winner based on mean
        mean_accs = [df['lr_test_acc'].mean(), df['rf_test_acc'].mean()]
        mean_labels = ['LR', 'RF']
        if has_cnn and len(cnn_data) > 0:
            mean_accs.append(cnn_data['cnn_test_acc'].mean())
            mean_labels.append('CNN')
        
        max_mean = max(mean_accs)
        winner_idx = mean_accs.index(max_mean)
        mean_row['Winner'] = mean_labels[winner_idx]
        
        table_data.append(mean_row)
        
        # Create DataFrame with requested column order
        desired_cols = ['Session', 'LR_Test_%', 'RF_Test_%', 'CNN_Test_%',
                        'LR_CV_%', 'RF_CV_%', 'CNN_CV_%',
                        'LR_F1', 'RF_F1', 'CNN_F1', 'Winner']
        table_df = pd.DataFrame(table_data)
        # Ensure all desired columns exist (fill missing with 'N/A')
        for c in desired_cols:
            if c not in table_df.columns:
                table_df[c] = 'N/A'
        table_df = table_df[desired_cols]

        # Save to Excel
        excel_path = os.path.join(self.output_dir, 'model_comparison_summary_table.xlsx')
        table_df.to_excel(excel_path, index=False, sheet_name='Model Comparison')
        print(f"Excel summary table saved: {excel_path}")

        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'model_comparison_summary_table.csv')
        table_df.to_csv(csv_path, index=False)
        print(f"CSV summary table saved: {csv_path}")
        
        # Save to TXT (nicely formatted) using the same column order as Excel/CSV
        txt_path = os.path.join(self.output_dir, 'model_comparison_summary_table.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*120 + "\n")
            f.write("MODEL PERFORMANCE COMPARISON - SUMMARY TABLE\n")
            f.write("="*120 + "\n\n")
            f.write("Table 1: Model performance comparison across all sessions\n\n")

            # Define column labels and widths
            header_labels = {
                'Session': 'Session',
                'LR_Test_%': 'LR Test %',
                'RF_Test_%': 'RF Test %',
                'CNN_Test_%': 'CNN Test %',
                'LR_CV_%': 'LR CV %',
                'RF_CV_%': 'RF CV %',
                'CNN_CV_%': 'CNN CV %',
                'LR_F1': 'LR F1',
                'RF_F1': 'RF F1',
                'CNN_F1': 'CNN F1',
                'Winner': 'Winner'
            }

            # Column widths for formatting
            col_widths = {
                'Session': 10,
                'LR_Test_%': 10,
                'RF_Test_%': 10,
                'CNN_Test_%': 10,
                'LR_CV_%': 10,
                'RF_CV_%': 10,
                'CNN_CV_%': 10,
                'LR_F1': 8,
                'RF_F1': 8,
                'CNN_F1': 8,
                'Winner': 10
            }

            # Write header
            header_items = [f"{header_labels[c]:<{col_widths[c]}}" for c in desired_cols]
            f.write(" ".join(header_items) + "\n")
            f.write("-"*120 + "\n")

            # Write data rows
            for _, r in table_df.iterrows():
                row_items = []
                for c in desired_cols:
                    val = r.get(c, 'N/A')
                    if pd.isna(val):
                        val = 'N/A'
                    row_items.append(f"{str(val):<{col_widths[c]}}")
                f.write(" ".join(row_items) + "\n")

            f.write("="*120 + "\n\n")

            # Add interpretation - dynamically generated based on actual results
            f.write("INTERPRETATION:\n\n")
            try:
                # Extract winner and accuracies from mean_row
                winner = mean_row.get('Winner', 'Unknown')
                winner_acc = mean_row.get(f'{winner}_Test_%', 'N/A')
                
                # Build comparison text
                f.write(f"{winner} achieved the highest mean test accuracy ({winner_acc}%)")
                
                # Add comparison with other models
                comparisons = []
                if winner != 'LR':
                    comparisons.append(f"Logistic Regression ({mean_row.get('LR_Test_%', 'N/A')}%)")
                if winner != 'RF':
                    comparisons.append(f"Random Forest ({mean_row.get('RF_Test_%', 'N/A')}%)")
                if has_cnn and winner != 'CNN' and mean_row.get('CNN_Test_%') not in (None, 'N/A'):
                    comparisons.append(f"CNN ({mean_row.get('CNN_Test_%')}%)")
                
                if comparisons:
                    f.write(f", outperforming {' and '.join(comparisons)}")
                f.write(".\n\n")
            except Exception as e:
                f.write(f"Summary statistics unavailable.\n\n")

            f.write("="*120 + "\n")

        print(f"TXT summary table saved: {txt_path}")
        print()

    def _determine_overall_winner(self, stats_summary, has_cnn):
        """
        Determine overall best model based on mean test accuracy.
        
        Args:
            stats_summary (dict): Statistics for each model
            has_cnn (bool): Whether CNN results are available
            
        Returns:
            str: Name of best performing model ('LR', 'RF', or 'CNN')
        """
        best_model = 'LR'
        best_acc = stats_summary['lr']['mean_acc']
        
        if stats_summary['rf']['mean_acc'] > best_acc:
            best_model = 'RF'
            best_acc = stats_summary['rf']['mean_acc']
        
        if has_cnn and stats_summary['cnn']['mean_acc'] > best_acc:
            best_model = 'CNN'
        
        return best_model

    def _save_detailed_results(self, df, stats_summary, statistical_tests, has_cnn):
        """
        Save detailed numerical results to CSV and JSON files.
        
        Args:
            df (pd.DataFrame): Session results dataframe
            stats_summary (dict): Calculated statistics
            statistical_tests (dict): Statistical test results
            has_cnn (bool): Whether CNN results are available
        """
        # Save session-by-session performance data
        csv_path = os.path.join(self.output_dir, 'three_model_performance.csv')
        df.to_csv(csv_path, index=False)
        print(f"Performance data saved: {csv_path}")
        
        # Save statistical tests as JSON
        import json
        json_path = os.path.join(self.output_dir, 'three_model_statistics.json')
        
        # Convert numpy and bool types to Python native types for JSON serialization
        def make_json_serializable(obj):
            """Convert numpy/bool types to JSON-serializable Python types."""
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return obj
        
        tests_serializable = {}
        for key, value in statistical_tests.items():
            tests_serializable[key] = {k: make_json_serializable(v) for k, v in value.items()}
        
        with open(json_path, 'w') as f:
            json.dump({
                'summary_statistics': {
                    model: {k: make_json_serializable(v) for k, v in stats.items()}
                    for model, stats in stats_summary.items()
                },
                'statistical_tests': tests_serializable
            }, f, indent=2)
        print(f"Statistical tests saved: {json_path}")

    def _print_summary(self, stats_summary, statistical_tests, has_cnn, total_sessions):
        """
        Print comprehensive summary to console and save to text file.
        
        Displays:
        - Overall performance for each model
        - Best performing model
        - Statistical significance of differences
        - Effect sizes (Cohen's d)
        
        Args:
            stats_summary (dict): Calculated statistics
            statistical_tests (dict): Statistical test results
            has_cnn (bool): Whether CNN results are available
            total_sessions (int): Total number of sessions
        """
        summary = []
        summary.append("\n" + "="*80)
        summary.append("THREE-MODEL COMPARISON SUMMARY")
        summary.append("="*80)
        
        summary.append("\nOVERALL PERFORMANCE:")
        summary.append(f"  Logistic Regression - Mean Test Accuracy: {stats_summary['lr']['mean_acc']*100:.2f}%")
        summary.append(f"  Random Forest       - Mean Test Accuracy: {stats_summary['rf']['mean_acc']*100:.2f}%")
        if has_cnn:
            summary.append(f"  CNN                 - Mean Test Accuracy: {stats_summary['cnn']['mean_acc']*100:.2f}%")
        
        # Determine winner
        best_model = self._determine_overall_winner(stats_summary, has_cnn)
        summary.append(f"\n  Best Model: {best_model}")
        
        summary.append("\nSTATISTICAL SIGNIFICANCE:")
        summary.append(f"  RF vs LR: {'Significant' if statistical_tests['lr_vs_rf']['significant'] else 'Not significant'} "
                      f"(p={statistical_tests['lr_vs_rf']['p_value']:.4f})")
        summary.append(f"  Effect size: {self._interpret_cohens_d(statistical_tests['lr_vs_rf']['cohens_d'])} "
                      f"(Cohen's d={statistical_tests['lr_vs_rf']['cohens_d']:.4f})")
        
        if has_cnn:
            summary.append(f"\n  CNN vs LR: {'Significant' if statistical_tests['lr_vs_cnn']['significant'] else 'Not significant'} "
                          f"(p={statistical_tests['lr_vs_cnn']['p_value']:.4f})")
            summary.append(f"  Effect size: {self._interpret_cohens_d(statistical_tests['lr_vs_cnn']['cohens_d'])} "
                          f"(Cohen's d={statistical_tests['lr_vs_cnn']['cohens_d']:.4f})")
            
            summary.append(f"\n  CNN vs RF: {'Significant' if statistical_tests['rf_vs_cnn']['significant'] else 'Not significant'} "
                          f"(p={statistical_tests['rf_vs_cnn']['p_value']:.4f})")
            summary.append(f"  Effect size: {self._interpret_cohens_d(statistical_tests['rf_vs_cnn']['cohens_d'])} "
                          f"(Cohen's d={statistical_tests['rf_vs_cnn']['cohens_d']:.4f})")
            
            if 'anova' in statistical_tests:
                summary.append(f"\n  ANOVA (all models): {'Significant' if statistical_tests['anova']['significant'] else 'Not significant'} "
                              f"(p={statistical_tests['anova']['p_value']:.4f})")
        
        # Add detailed statistical test results (paired t-test, Wilcoxon)
        summary.append("\n" + "="*80)
        summary.append("DETAILED STATISTICAL TEST RESULTS (RF vs LR)")
        summary.append("="*80)
        summary.append("\nPaired t-test:")
        summary.append(f"  t-statistic: {statistical_tests['lr_vs_rf']['t_statistic']:.4f}")
        summary.append(f"  p-value: {statistical_tests['lr_vs_rf']['p_value']:.4f}")
        summary.append(f"  Significant: {'YES' if statistical_tests['lr_vs_rf']['significant'] else 'NO'} (α=0.05)")
        
        # Calculate additional metrics for RF vs LR
        summary.append(f"\nEffect size (Cohen's d): {statistical_tests['lr_vs_rf']['cohens_d']:.4f}")
        summary.append(f"  Interpretation: {self._interpret_cohens_d(statistical_tests['lr_vs_rf']['cohens_d'])}")
        
        summary.append("="*80)
        
        # Print to console
        summary_text = "\n".join(summary)
        print(summary_text)
        
        # Save to file
        txt_path = os.path.join(self.output_dir, 'three_model_summary.txt')
        with open(txt_path, 'w') as f:
            f.write(summary_text)
        print(f"Summary saved: {txt_path}")

    def _interpret_cohens_d(self, d):
        """
        Interpret Cohen's d effect size.
        
        Provides standard interpretation thresholds:
        - |d| < 0.2: Negligible effect
        - 0.2 ≤ |d| < 0.5: Small effect
        - 0.5 ≤ |d| < 0.8: Medium effect
        - |d| ≥ 0.8: Large effect
        
        Args:
            d (float): Cohen's d value
            
        Returns:
            str: Interpretation of effect size
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"


# Example usage
if __name__ == "__main__":
    print("ThreeModelComparator Module")
    print("Use this class to compare LR, RF, and CNN results across sessions.")
    print("\nExample usage:")
    print("  comparator = ThreeModelComparator(output_dir='final_results')")
    print("  comparator.add_session_results('50-1', lr_metrics, rf_metrics, cnn_metrics)")
    print("  comparator.generate_comprehensive_comparison()")
