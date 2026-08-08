#!/usr/bin/env python3
"""
📊 Pareto Analysis Tools for HivePath AI
========================================
Advanced analysis tools for Pareto frontier optimization results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import json
from datetime import datetime

class ParetoAnalysisTools:
    """Advanced analysis tools for Pareto frontier results"""
    
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def load_pareto_results(self, file_path: str) -> Dict[str, Any]:
        """Load Pareto optimization results from JSON file"""
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_pareto_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert Pareto results to pandas DataFrame for analysis"""
        
        pareto_solutions = results['results']['pareto_frontier']
        
        data = []
        for i, solution in enumerate(pareto_solutions):
            row = {
                'solution_id': i,
                'crowding_distance': solution['crowding_distance']
            }
            
            # Add objectives
            for obj_name, value in solution['objectives'].items():
                row[f'obj_{obj_name}'] = value
            
            # Add variables
            for var_name, value in solution['variables'].items():
                row[f'var_{var_name}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_pareto_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced Pareto frontier metrics"""
        
        metrics = {}
        
        # Hypervolume (approximation)
        obj_cols = [col for col in df.columns if col.startswith('obj_')]
        if len(obj_cols) >= 2:
            # Use 2D hypervolume approximation
            obj1_values = df[obj_cols[0]].values
            obj2_values = df[obj_cols[1]].values
            
            # Normalize values
            obj1_norm = (obj1_values - obj1_values.min()) / (obj1_values.max() - obj1_values.min())
            obj2_norm = (obj2_values - obj2_values.min()) / (obj2_values.max() - obj2_values.min())
            
            # Calculate area under Pareto curve
            sorted_indices = np.argsort(obj1_norm)
            sorted_obj1 = obj1_norm[sorted_indices]
            sorted_obj2 = obj2_norm[sorted_indices]
            
            hypervolume = 0
            for i in range(len(sorted_obj1) - 1):
                width = sorted_obj1[i + 1] - sorted_obj1[i]
                height = sorted_obj2[i]
                hypervolume += width * height
            
            metrics['hypervolume_2d'] = hypervolume
        
        # Spread (diversity metric)
        if len(obj_cols) >= 2:
            distances = []
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    dist = np.sqrt(sum((df.iloc[i][col] - df.iloc[j][col])**2 for col in obj_cols))
                    distances.append(dist)
            
            metrics['spread'] = np.std(distances) if distances else 0
            metrics['min_distance'] = min(distances) if distances else 0
            metrics['max_distance'] = max(distances) if distances else 0
        
        # Crowding distance statistics
        metrics['avg_crowding_distance'] = df['crowding_distance'].mean()
        metrics['max_crowding_distance'] = df['crowding_distance'].max()
        metrics['crowding_distance_std'] = df['crowding_distance'].std()
        
        return metrics
    
    def cluster_pareto_solutions(self, df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster Pareto solutions to identify solution groups"""
        
        obj_cols = [col for col in df.columns if col.startswith('obj_')]
        
        if len(obj_cols) < 2:
            return {'error': 'Need at least 2 objectives for clustering'}
        
        # Prepare data for clustering
        X = df[obj_cols].values
        
        # Calculate distance matrix
        distances = pdist(X, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        # Get cluster labels
        from scipy.cluster.hierarchy import fcluster
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Add cluster labels to dataframe
        df['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(1, n_clusters + 1):
            cluster_data = df[df['cluster'] == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_objectives': {col: cluster_data[col].mean() for col in obj_cols},
                'solution_ids': cluster_data['solution_id'].tolist()
            }
        
        return {
            'cluster_labels': cluster_labels,
            'linkage_matrix': linkage_matrix,
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters
        }
    
    def visualize_pareto_analysis(self, df: pd.DataFrame, cluster_results: Dict[str, Any] = None, 
                                save_path: str = None) -> None:
        """Create comprehensive Pareto analysis visualizations"""
        
        obj_cols = [col for col in df.columns if col.startswith('obj_')]
        var_cols = [col for col in df.columns if col.startswith('var_')]
        
        # Create figure with subplots
        n_plots = 4
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 HivePath AI Pareto Frontier Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Pareto frontier with clustering
        ax1 = axes[0, 0]
        if len(obj_cols) >= 2 and cluster_results:
            colors = self.colors[:cluster_results['n_clusters']]
            for cluster_id in range(1, cluster_results['n_clusters'] + 1):
                cluster_data = df[df['cluster'] == cluster_id]
                ax1.scatter(cluster_data[obj_cols[0]], cluster_data[obj_cols[1]], 
                           c=colors[cluster_id-1], label=f'Cluster {cluster_id}', 
                           s=60, alpha=0.7)
        else:
            ax1.scatter(df[obj_cols[0]], df[obj_cols[1]], c='red', s=60, alpha=0.7)
        
        ax1.set_xlabel(obj_cols[0].replace('obj_', '').title())
        ax1.set_ylabel(obj_cols[1].replace('obj_', '').title())
        ax1.set_title('Pareto Frontier with Clustering')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Crowding distance distribution
        ax2 = axes[0, 1]
        ax2.hist(df['crowding_distance'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(df['crowding_distance'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["crowding_distance"].mean():.3f}')
        ax2.set_xlabel('Crowding Distance')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Crowding Distance Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Objective correlation heatmap
        ax3 = axes[1, 0]
        if len(obj_cols) > 2:
            corr_matrix = df[obj_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax3, cbar_kws={'shrink': 0.8})
            ax3.set_title('Objective Correlation Matrix')
        else:
            ax3.text(0.5, 0.5, 'Need 3+ objectives\nfor correlation matrix', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Objective Correlation Matrix')
        
        # Plot 4: Variable importance
        ax4 = axes[1, 1]
        if var_cols:
            # Calculate variance for each variable
            var_importance = df[var_cols].var().sort_values(ascending=True)
            ax4.barh(range(len(var_importance)), var_importance.values, 
                    color='lightgreen', alpha=0.7)
            ax4.set_yticks(range(len(var_importance)))
            ax4.set_yticklabels([col.replace('var_', '') for col in var_importance.index])
            ax4.set_xlabel('Variance')
            ax4.set_title('Variable Importance (Variance)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No variables found\nfor importance analysis', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Variable Importance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_pareto_report(self, results: Dict[str, Any], analysis: Dict[str, Any] = None) -> str:
        """Generate comprehensive Pareto analysis report"""
        
        pareto_solutions = results['results']['pareto_frontier']
        optimization_params = results['results']['optimization_parameters']
        
        report = []
        report.append("🎯 HIVEPATH AI - PARETO FRONTIER ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Optimization Summary
        report.append("📊 OPTIMIZATION SUMMARY")
        report.append("-" * 30)
        report.append(f"Generations: {optimization_params['num_generations']}")
        report.append(f"Population Size: {optimization_params['population_size']}")
        report.append(f"Objectives: {', '.join(optimization_params['objectives'])}")
        report.append(f"Pareto Solutions: {len(pareto_solutions)}")
        report.append("")
        
        # Objective Analysis
        if analysis and 'objective_statistics' in analysis:
            report.append("🎯 OBJECTIVE ANALYSIS")
            report.append("-" * 30)
            
            for obj_name, stats in analysis['objective_statistics'].items():
                report.append(f"{obj_name.title()}:")
                report.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f}")
                report.append(f"  Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                report.append("")
        
        # Best Solutions
        if analysis and 'best_solutions' in analysis:
            report.append("🏆 BEST SOLUTIONS BY OBJECTIVE")
            report.append("-" * 30)
            
            for obj_name, best_sol in analysis['best_solutions'].items():
                report.append(f"Best {obj_name.title()}:")
                report.append(f"  Solution ID: {best_sol['solution_index']}")
                report.append(f"  Value: {best_sol['objectives'][obj_name]:.2f}")
                report.append("")
        
        # Top 5 Pareto Solutions
        report.append("🥇 TOP 5 PARETO SOLUTIONS")
        report.append("-" * 30)
        
        for i, solution in enumerate(pareto_solutions[:5]):
            report.append(f"Solution {i+1}:")
            report.append("  Objectives:")
            for obj_name, value in solution['objectives'].items():
                report.append(f"    {obj_name}: {value:.2f}")
            report.append("  Variables:")
            for var_name, value in solution['variables'].items():
                report.append(f"    {var_name}: {value:.2f}")
            report.append(f"  Crowding Distance: {solution['crowding_distance']:.3f}")
            report.append("")
        
        return "\n".join(report)
    
    def export_pareto_solutions(self, results: Dict[str, Any], format: str = 'csv', 
                              file_path: str = None) -> str:
        """Export Pareto solutions to various formats"""
        
        df = self.create_pareto_dataframe(results)
        
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"pareto_solutions_{timestamp}.{format}"
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📁 Pareto solutions exported to: {file_path}")
        return file_path

def main():
    """Main function to demonstrate Pareto analysis tools"""
    
    print("📊 HivePath AI - Pareto Analysis Tools")
    print("=" * 50)
    
    # Initialize analysis tools
    analyzer = ParetoAnalysisTools()
    
    # Example: Load and analyze results
    print("🔍 Loading Pareto optimization results...")
    
    # For demonstration, create sample results
    sample_results = {
        'results': {
            'pareto_frontier': [
                {
                    'objectives': {'cost': 100, 'time': 120, 'risk': 0.3, 'accessibility': -5, 'environmental': 25},
                    'variables': {'distance': 50, 'time': 100, 'vehicle_count': 2, 'accessibility_score': 80, 'risk_score': 0.3},
                    'crowding_distance': 0.5
                },
                {
                    'objectives': {'cost': 120, 'time': 100, 'risk': 0.2, 'accessibility': -3, 'environmental': 30},
                    'variables': {'distance': 60, 'time': 80, 'vehicle_count': 3, 'accessibility_score': 85, 'risk_score': 0.2},
                    'crowding_distance': 0.4
                }
            ],
            'optimization_parameters': {
                'num_generations': 30,
                'population_size': 80,
                'objectives': ['cost', 'time', 'risk', 'accessibility', 'environmental']
            }
        }
    }
    
    # Create DataFrame
    df = analyzer.create_pareto_dataframe(sample_results)
    print(f"📊 Created DataFrame with {len(df)} solutions")
    
    # Calculate metrics
    metrics = analyzer.calculate_pareto_metrics(df)
    print(f"📈 Calculated {len(metrics)} Pareto metrics")
    
    # Cluster solutions
    cluster_results = analyzer.cluster_pareto_solutions(df, n_clusters=2)
    print(f"🔍 Clustered solutions into {cluster_results['n_clusters']} groups")
    
    # Generate report
    report = analyzer.generate_pareto_report(sample_results)
    print("\n" + report)
    
    # Export solutions
    csv_path = analyzer.export_pareto_solutions(sample_results, 'csv')
    
    print(f"\n✅ Pareto analysis tools demonstration complete!")
    print(f"📁 Results exported to: {csv_path}")

if __name__ == "__main__":
    main()
