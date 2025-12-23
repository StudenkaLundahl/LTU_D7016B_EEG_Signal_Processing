"""
Results Analyzer Module
Generates comprehensive analysis tables, statistics, and comparison plots
for EEG classification results across all sessions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json

class ResultsAnalyzer:
    """
    Comprehensive analyzer for EEG classification results.
    
    Generates:
    - Session-by-session performance tables
    - Model comparison plots (LR vs RF)
    - Statistical significance tests
    - Environmental factor analysis
    - Grand average PSD plots
    """
    
    def __init__(self, output_dir='analysis_results'):
        """
        Initialize analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for results
        self.results = {
            'sessions': [],
            'lr_metrics': [],
            'rf_metrics': [],
            'behavioral': [],
            'environmental': []
        }
        
    def add_session_results(self, session_name, lr_metrics, rf_metrics, 
                           behavioral_hits, environmental_notes=''):
        """
        Add results from a single session.
        
        Args:
            session_name (str): Session identifier (e.g., '50-1')
            lr_metrics (dict): LR metrics from evaluate_model()
            rf_metrics (dict): RF metrics from evaluate_model()
            behavioral_hits (int): Number of successful trials
            environmental_notes (str): Notes about data quality
        """
        self.results['sessions'].append(session_name)
        self.results['lr_metrics'].append(lr_metrics)
        self.results['rf_metrics'].append(rf_metrics)
        self.results['behavioral'].append(behavioral_hits)
        self.results['environmental'].append(environmental_notes)
        
    def generate_performance_table(self):
        """
        Generate session-by-session performance comparison table.
        
        Returns:
            pd.DataFrame: Comprehensive performance table
        """
        data = []
        
        for i, session in enumerate(self.results['sessions']):
            lr = self.results['lr_metrics'][i]
            rf = self.results['rf_metrics'][i]
            hits = self.results['behavioral'][i]
            
            # Determine winner
            if rf['test_accuracy'] > lr['test_accuracy']:
                winner = 'RF'
                improvement = (rf['test_accuracy'] - lr['test_accuracy']) * 100
            elif lr['test_accuracy'] > rf['test_accuracy']:
                winner = 'LR'
                improvement = (lr['test_accuracy'] - rf['test_accuracy']) * 100
            else:
                winner = 'Tie'
                improvement = 0
            
            data.append({
                'Session': session,
                'Behavioral_Hits': hits,
                'Hit_Rate_%': (hits / 50) * 100,
                'LR_Test_Acc_%': lr['test_accuracy'] * 100,
                'RF_Test_Acc_%': rf['test_accuracy'] * 100,
                'LR_CV_Acc_%': lr.get('cv_mean', 0) * 100,
                'RF_CV_Acc_%': rf.get('cv_mean', 0) * 100,
                'LR_F1': lr['f1_score'],
                'RF_F1': rf['f1_score'],
                'Winner': winner,
                'Improvement_%': abs(improvement),
                'Environmental': self.results['environmental'][i]
            })
        
        df = pd.DataFrame(data)
        
        # Add summary row
        summary = {
            'Session': 'MEAN',
            'Behavioral_Hits': df['Behavioral_Hits'].mean(),
            'Hit_Rate_%': df['Hit_Rate_%'].mean(),
            'LR_Test_Acc_%': df['LR_Test_Acc_%'].mean(),
            'RF_Test_Acc_%': df['RF_Test_Acc_%'].mean(),
            'LR_CV_Acc_%': df['LR_CV_Acc_%'].mean(),
            'RF_CV_Acc_%': df['RF_CV_Acc_%'].mean(),
            'LR_F1': df['LR_F1'].mean(),
            'RF_F1': df['RF_F1'].mean(),
            'Winner': self._determine_overall_winner(df),
            'Improvement_%': df['Improvement_%'].mean(),
            'Environmental': 'All sessions'
        }
        
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        
        # Save to CSV
        df.to_csv(self.output_dir / 'performance_comparison.csv', index=False)
        print(f"Saved performance table: {self.output_dir / 'performance_comparison.csv'}")
        
        return df
    
    def _determine_overall_winner(self, df):
        """Determine overall winner based on wins and average performance."""
        rf_wins = (df['Winner'] == 'RF').sum()
        lr_wins = (df['Winner'] == 'LR').sum()
        
        if rf_wins > lr_wins:
            return 'RF'
        elif lr_wins > rf_wins:
            return 'LR'
        else:
            # Check average accuracy
            if df['RF_Test_Acc_%'].mean() > df['LR_Test_Acc_%'].mean():
                return 'RF (by avg)'
            else:
                return 'LR (by avg)'
    
    def plot_model_comparison(self):
        """
        Create comprehensive model comparison plots.
        """
        sessions = self.results['sessions']
        
        # Extract accuracies
        lr_test = [m['test_accuracy'] * 100 for m in self.results['lr_metrics']]
        rf_test = [m['test_accuracy'] * 100 for m in self.results['rf_metrics']]
        lr_cv = [m.get('cv_mean', 0) * 100 for m in self.results['lr_metrics']]
        rf_cv = [m.get('cv_mean', 0) * 100 for m in self.results['rf_metrics']]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        
        # 1. Test Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(sessions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, lr_test, width, label='Logistic Regression', 
                    color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, rf_test, width, label='Random Forest', 
                    color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax1.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Chance Level')
        ax1.set_xlabel('Session', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Test Accuracy: Logistic Regression vs Random Forest', 
                    fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sessions, rotation=0)
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # 2. Cross-Validation Accuracy
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(sessions, lr_cv, marker='o', linewidth=2, markersize=8, 
                label='LR CV', color='#3498db')
        ax2.plot(sessions, rf_cv, marker='s', linewidth=2, markersize=8, 
                label='RF CV', color='#e74c3c')
        ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.set_ylabel('CV Accuracy (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Cross-Validation Accuracy', fontsize=12, fontweight='bold', pad=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. F1-Score Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        lr_f1 = [m['f1_score'] for m in self.results['lr_metrics']]
        rf_f1 = [m['f1_score'] for m in self.results['rf_metrics']]
        
        # Color points based on which model performs better
        colors_f1 = []
        for i in range(len(lr_f1)):
            if rf_f1[i] > lr_f1[i]:
                colors_f1.append('#e74c3c')  # RF wins - red
            elif lr_f1[i] > rf_f1[i]:
                colors_f1.append('#3498db')  # LR wins - blue
            else:
                colors_f1.append('#95a5a6')  # Tie - grey
        
        ax3.scatter(lr_f1, rf_f1, s=150, c=colors_f1, alpha=0.7, edgecolors='black', linewidth=2)
        for i, session in enumerate(sessions):
            ax3.annotate(session, (lr_f1[i], rf_f1[i]), fontsize=9, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Add diagonal line (equal performance)
        lims = [0, 1]
        ax3.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Equal Performance')
        ax3.set_xlabel('LR F1-Score', fontsize=11, fontweight='bold')
        ax3.set_ylabel('RF F1-Score', fontsize=11, fontweight='bold')
        ax3.set_title('F1-Score: LR vs RF', fontsize=12, fontweight='bold', pad=10)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        
        # 4. Winner Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        winners = []
        for i in range(len(sessions)):
            if rf_test[i] > lr_test[i]:
                winners.append('RF')
            elif lr_test[i] > rf_test[i]:
                winners.append('LR')
            else:
                winners.append('Tie')
        
        from collections import Counter
        winner_counts = Counter(winners)
        colors_winner = {'RF': '#e74c3c', 'LR': '#3498db', 'Tie': '#95a5a6'}
        
        bars = ax4.bar(winner_counts.keys(), winner_counts.values(), 
                    color=[colors_winner[k] for k in winner_counts.keys()],
                    edgecolor='black', linewidth=2, alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax4.set_ylabel('Number of Sessions', fontsize=11, fontweight='bold')
        ax4.set_title('Model Comparison: Who Wins?', fontsize=12, fontweight='bold', pad=10)
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Accuracy Improvement Distribution
        ax5 = fig.add_subplot(gs[2, 1])
        improvements = np.array(rf_test) - np.array(lr_test)
        
        # Green when RF wins (positive), Red when LR wins (negative), Grey for tie
        colors_bars = ['#27ae60' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' 
                    for x in improvements]
        bars = ax5.bar(sessions, improvements, color=colors_bars, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax5.set_xlabel('Session', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Accuracy Difference (RF - LR) %', fontsize=11, fontweight='bold')
        ax5.set_title('RF Improvement over LR', fontsize=12, fontweight='bold', pad=10)
        ax5.grid(axis='y', alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison_comprehensive.png', 
                dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot: {self.output_dir / 'model_comparison_comprehensive.png'}")
        plt.close()
    
    def perform_statistical_tests(self):
        """
        Perform statistical significance tests on model performance.
        
        Returns:
            dict: Statistical test results
        """
        lr_test = [m['test_accuracy'] for m in self.results['lr_metrics']]
        rf_test = [m['test_accuracy'] for m in self.results['rf_metrics']]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rf_test, lr_test)
        
        # Wilcoxon signed-rank test (non-parametric)
        w_stat, w_pvalue = stats.wilcoxon(rf_test, lr_test)
        
        # Effect size (Cohen's d)
        diff = np.array(rf_test) - np.array(lr_test)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Convert numpy types to Python native types for JSON serialization
        results = {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            },
            'wilcoxon_test': {
                'w_statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant': bool(w_pvalue < 0.05)
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'mean_difference': float(np.mean(diff) * 100),
            'std_difference': float(np.std(diff) * 100)
        }
        
        # Print summary
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TEST RESULTS")
        print("="*70)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")
        
        print(f"\nWilcoxon signed-rank test:")
        print(f"  W-statistic: {w_stat:.4f}")
        print(f"  p-value: {w_pvalue:.4f}")
        print(f"  Significant: {'YES' if w_pvalue < 0.05 else 'NO'} (α=0.05)")
        
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        print(f"  Interpretation: {self._interpret_cohens_d(cohens_d)}")
        
        print(f"\nMean accuracy difference (RF - LR): {np.mean(diff)*100:.2f}%")
        print(f"Standard deviation: {np.std(diff)*100:.2f}%")
        print("="*70 + "\n")
        
        # Save to file
        with open(self.output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def create_environmental_impact_table(self):
        """
        Create environmental factor impact analysis table.
        
        Returns:
            pd.DataFrame: Environmental impact table
        """
        data = []
        
        for i, session in enumerate(self.results['sessions']):
            hits = self.results['behavioral'][i]
            hit_rate = (hits / 50) * 100
            
            lr_acc = self.results['lr_metrics'][i]['test_accuracy'] * 100
            rf_acc = self.results['rf_metrics'][i]['test_accuracy'] * 100
            avg_acc = (lr_acc + rf_acc) / 2
            
            env_note = self.results['environmental'][i]
            
            data.append({
                'Session': session,
                'Behavioral_Hits': hits,
                'Hit_Rate_%': hit_rate,
                'Avg_Classification_%': avg_acc,
                'Environmental_Factor': env_note
            })
        
        df = pd.DataFrame(data)
        
        # Sort by hit rate for better visualization
        df_sorted = df.sort_values('Hit_Rate_%', ascending=False)
        
        # Save to CSV
        df_sorted.to_csv(self.output_dir / 'environmental_impact.csv', index=False)
        print(f"Saved environmental impact table: {self.output_dir / 'environmental_impact.csv'}")
        
        # Create visualization
        self._plot_environmental_impact(df_sorted)
        
        return df_sorted
    
    def _plot_environmental_impact(self, df):
        """Plot environmental factor impact."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Hit rate vs Classification accuracy
        colors = []
        for env in df['Environmental_Factor']:
            if 'Optimal' in env or 'Good' in env or env == '':
                colors.append('#27ae60')
            elif 'Sunlight' in env or 'Hunger' in env:
                colors.append('#e74c3c')
            else:
                colors.append('#3498db')
        
        ax1.scatter(df['Hit_Rate_%'], df['Avg_Classification_%'], 
                   s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
        
        for i, row in df.iterrows():
            ax1.annotate(row['Session'], 
                        (row['Hit_Rate_%'], row['Avg_Classification_%']),
                        fontsize=9, xytext=(5, 5), textcoords='offset points')
        
        # Add correlation line
        z = np.polyfit(df['Hit_Rate_%'], df['Avg_Classification_%'], 1)
        p = np.poly1d(z)
        ax1.plot(df['Hit_Rate_%'], p(df['Hit_Rate_%']), 
                "k--", alpha=0.5, linewidth=2, label=f'Trend line')
        
        # Calculate correlation
        corr = df['Hit_Rate_%'].corr(df['Avg_Classification_%'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Behavioral Hit Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Classification Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Behavioral Performance vs Classification Accuracy', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Environmental conditions
        env_categories = df['Environmental_Factor'].value_counts()
        colors_env = ['#27ae60' if 'Optimal' in c or 'Good' in c or c == '' 
                     else '#e74c3c' for c in env_categories.index]
        
        bars = ax2.barh(range(len(env_categories)), env_categories.values, 
                       color=colors_env, edgecolor='black', linewidth=2, alpha=0.8)
        ax2.set_yticks(range(len(env_categories)))
        ax2.set_yticklabels(env_categories.index, fontsize=10)
        ax2.set_xlabel('Number of Sessions', fontsize=12, fontweight='bold')
        ax2.set_title('Environmental Conditions Distribution', 
                     fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', 
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'environmental_impact.png', 
                   dpi=300, bbox_inches='tight')
        print(f"Saved environmental impact plot: {self.output_dir / 'environmental_impact.png'}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        # Generate all components
        perf_table = self.generate_performance_table()
        self.plot_model_comparison()
        stats_results = self.perform_statistical_tests()
        env_table = self.create_environmental_impact_table()
        
        # Create text summary
        summary_text = []
        summary_text.append("="*80)
        summary_text.append("COMPREHENSIVE ANALYSIS SUMMARY")
        summary_text.append("="*80)
        summary_text.append("")
        
        # Overall performance
        summary_text.append("OVERALL PERFORMANCE:")
        summary_text.append(f"  Logistic Regression - Mean Test Accuracy: {perf_table.loc[perf_table['Session']=='MEAN', 'LR_Test_Acc_%'].values[0]:.2f}%")
        summary_text.append(f"  Random Forest       - Mean Test Accuracy: {perf_table.loc[perf_table['Session']=='MEAN', 'RF_Test_Acc_%'].values[0]:.2f}%")
        summary_text.append(f"  Winner: {perf_table.loc[perf_table['Session']=='MEAN', 'Winner'].values[0]}")
        summary_text.append("")
        
        # Statistical significance
        summary_text.append("STATISTICAL SIGNIFICANCE:")
        summary_text.append(f"  RF vs LR: {'Significant' if stats_results['paired_t_test']['significant'] else 'Not significant'} (p={stats_results['paired_t_test']['p_value']:.4f})")
        summary_text.append(f"  Effect size: {stats_results['effect_size']['interpretation']} (Cohen's d={stats_results['effect_size']['cohens_d']:.4f})")
        summary_text.append("")
        
        # Environmental impact
        best_session = env_table.iloc[0]
        worst_session = env_table.iloc[-1]
        summary_text.append("ENVIRONMENTAL IMPACT:")
        summary_text.append(f"  Best session: {best_session['Session']} ({best_session['Hit_Rate_%']:.1f}% hits, {best_session['Environmental_Factor']})")
        summary_text.append(f"  Worst session: {worst_session['Session']} ({worst_session['Hit_Rate_%']:.1f}% hits, {worst_session['Environmental_Factor']})")
        summary_text.append(f"  Correlation (behavior vs classification): r={env_table['Hit_Rate_%'].corr(env_table['Avg_Classification_%']):.3f}")
        summary_text.append("")
        
        summary_text.append("="*80)
        
        # Print and save
        summary = "\n".join(summary_text)
        print(summary)
        
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(summary)
        
        print(f"\nAll results saved to: {self.output_dir}/")
        print("  - performance_comparison.csv")
        print("  - model_comparison_comprehensive.png")
        print("  - statistical_tests.json")
        print("  - environmental_impact.csv")
        print("  - environmental_impact.png")
        print("  - analysis_summary.txt")
