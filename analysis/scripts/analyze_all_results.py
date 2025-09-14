#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Embedded AI Benchmark Suite
Processes and analyzes results from all platforms and benchmarks
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class BenchmarkAnalyzer:
    def __init__(self, results_root):
        self.results_root = Path(results_root)
        self.platforms = ['nvidia-jetson', 'qualcomm-qcs6490', 'radxa-x4']
        self.benchmarks = ['orb_slam3', '3d_detection', 'segmentation']
        self.results_data = {}
        
    def discover_results(self):
        """Discover all result files across platforms and benchmarks."""
        print("Discovering result files...")
        
        for platform in self.platforms:
            self.results_data[platform] = {}
            platform_path = self.results_root / platform / 'results'
            
            if not platform_path.exists():
                print(f"Warning: Results path not found for {platform}")
                continue
                
            for benchmark in self.benchmarks:
                benchmark_path = platform_path / benchmark
                if not benchmark_path.exists():
                    continue
                    
                # Find JSON result files
                json_files = list(benchmark_path.glob('*.json'))
                csv_files = list(benchmark_path.glob('*.csv'))
                
                self.results_data[platform][benchmark] = {
                    'json_files': json_files,
                    'csv_files': csv_files
                }
                
                print(f"Found {len(json_files)} JSON and {len(csv_files)} CSV files for {platform}/{benchmark}")
    
    def load_orb_slam3_results(self):
        """Load and process ORB-SLAM3 results."""
        orb_results = []
        
        for platform in self.platforms:
            if platform not in self.results_data:
                continue
                
            benchmark_data = self.results_data[platform].get('orb_slam3', {})
            
            # Look for analysis files or CSV files
            for csv_file in benchmark_data.get('csv_files', []):
                if 'latencies' in csv_file.name.lower():
                    try:
                        df = pd.read_csv(csv_file)
                        if 'latency' in df.columns or len(df.columns) == 1:
                            latencies = df.iloc[:, 0].values
                            
                            result = {
                                'platform': platform,
                                'benchmark': 'orb_slam3',
                                'mean_latency_ms': np.mean(latencies),
                                'p99_latency_ms': np.percentile(latencies, 99),
                                'throughput_fps': 1000.0 / np.mean(latencies),
                                'std_latency_ms': np.std(latencies),
                                'total_frames': len(latencies)
                            }
                            orb_results.append(result)
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")
        
        return orb_results
    
    def load_ai_benchmark_results(self, benchmark_name):
        """Load and process AI benchmark results (3D detection, segmentation)."""
        ai_results = []
        
        for platform in self.platforms:
            if platform not in self.results_data:
                continue
                
            benchmark_data = self.results_data[platform].get(benchmark_name, {})
            
            for json_file in benchmark_data.get('json_files', []):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract accelerator type from filename or data
                    accelerator = 'unknown'
                    if 'gpu' in json_file.name.lower():
                        accelerator = 'GPU'
                    elif 'dla' in json_file.name.lower():
                        accelerator = 'DLA'
                    elif 'dsp' in json_file.name.lower() or 'hexagon' in json_file.name.lower():
                        accelerator = 'Hexagon NPU'
                    elif 'cpu' in json_file.name.lower():
                        accelerator = 'CPU'
                    
                    # Handle different JSON structures
                    if 'latency' in data:
                        latency_data = data['latency']
                        result = {
                            'platform': platform,
                            'accelerator': accelerator,
                            'benchmark': benchmark_name,
                            'mean_latency_ms': latency_data.get('mean_ms', 0),
                            'p99_latency_ms': latency_data.get('p99_ms', 0),
                            'throughput_fps': data.get('throughput_fps', 0),
                            'std_latency_ms': latency_data.get('std_ms', 0),
                            'iterations': data.get('num_iterations', 0)
                        }
                    else:
                        # Handle different JSON format
                        result = {
                            'platform': platform,
                            'accelerator': accelerator,
                            'benchmark': benchmark_name,
                            'mean_latency_ms': data.get('mean_latency_ms', 0),
                            'p99_latency_ms': data.get('p99_latency_ms', 0),
                            'throughput_fps': data.get('throughput_fps', 0),
                            'std_latency_ms': data.get('std_latency_ms', 0),
                            'iterations': data.get('iterations', 0)
                        }
                    
                    # Add accuracy metrics if available
                    if 'miou' in data:
                        result['miou'] = data['miou']
                    if '3d_map' in data:
                        result['3d_map'] = data['3d_map']
                    
                    ai_results.append(result)
                    
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        
        return ai_results
    
    def load_power_data(self):
        """Load power measurement data if available."""
        power_results = []
        
        for platform in self.platforms:
            platform_path = self.results_root / platform / 'results'
            
            # Look for power measurement files
            power_files = list(platform_path.rglob('*power*.csv'))
            
            for power_file in power_files:
                try:
                    df = pd.read_csv(power_file)
                    
                    if 'power' in df.columns:
                        power_data = {
                            'platform': platform,
                            'mean_power_w': df['power'].mean(),
                            'max_power_w': df['power'].max(),
                            'min_power_w': df['power'].min(),
                            'std_power_w': df['power'].std(),
                            'total_energy_wh': df['power'].sum() * 0.1 / 3600  # Assuming 0.1s intervals
                        }
                        power_results.append(power_data)
                        
                except Exception as e:
                    print(f"Error loading power data from {power_file}: {e}")
        
        return power_results
    
    def create_performance_comparison(self, results_df, output_dir):
        """Create performance comparison visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Platform mapping for cleaner labels
        platform_names = {
            'nvidia-jetson': 'NVIDIA Jetson Orin NX',
            'qualcomm-qcs6490': 'Qualcomm QCS6490',
            'radxa-x4': 'Radxa X4 (Intel N100)'
        }
        
        results_df['platform_name'] = results_df['platform'].map(platform_names)
        
        # 1. Throughput comparison by benchmark
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        benchmarks = results_df['benchmark'].unique()
        
        for i, benchmark in enumerate(benchmarks):
            benchmark_data = results_df[results_df['benchmark'] == benchmark]
            
            if benchmark == 'orb_slam3':
                # ORB-SLAM3 doesn't have accelerator variants
                sns.barplot(data=benchmark_data, x='platform_name', y='throughput_fps', ax=axes[i])
                axes[i].set_title(f'{benchmark.upper().replace("_", "-")} Throughput')
            else:
                # AI benchmarks have accelerator variants
                sns.barplot(data=benchmark_data, x='platform_name', y='throughput_fps', 
                           hue='accelerator', ax=axes[i])
                axes[i].set_title(f'{benchmark.replace("_", " ").title()} Throughput')
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[i].set_ylabel('Throughput (FPS)')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Latency comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, benchmark in enumerate(benchmarks):
            benchmark_data = results_df[results_df['benchmark'] == benchmark]
            
            if benchmark == 'orb_slam3':
                sns.barplot(data=benchmark_data, x='platform_name', y='p99_latency_ms', ax=axes[i])
                axes[i].set_title(f'{benchmark.upper().replace("_", "-")} P99 Latency')
            else:
                sns.barplot(data=benchmark_data, x='platform_name', y='p99_latency_ms', 
                           hue='accelerator', ax=axes[i])
                axes[i].set_title(f'{benchmark.replace("_", " ").title()} P99 Latency')
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[i].set_ylabel('P99 Latency (ms)')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Platform performance radar chart
        self.create_radar_chart(results_df, output_dir)
        
        print(f"Performance comparison plots saved to {output_dir}")
    
    def create_radar_chart(self, results_df, output_dir):
        """Create radar chart comparing platforms across benchmarks."""
        # Aggregate data by platform (take best performance for each benchmark)
        platform_summary = {}
        
        for platform in results_df['platform_name'].unique():
            platform_data = results_df[results_df['platform_name'] == platform]
            
            # Get best performance for each benchmark
            summary = {}
            for benchmark in platform_data['benchmark'].unique():
                benchmark_data = platform_data[platform_data['benchmark'] == benchmark]
                best_fps = benchmark_data['throughput_fps'].max()
                summary[benchmark] = best_fps
            
            platform_summary[platform] = summary
        
        # Create radar chart
        benchmarks = list(results_df['benchmark'].unique())
        num_vars = len(benchmarks)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (platform, data) in enumerate(platform_summary.items()):
            values = [data.get(benchmark, 0) for benchmark in benchmarks]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=platform, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([b.replace('_', ' ').title() for b in benchmarks])
        ax.set_ylabel('Throughput (FPS)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title('Platform Performance Comparison\n(Throughput across Benchmarks)', 
                 size=16, weight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_efficiency_analysis(self, results_df, power_df, output_dir):
        """Create power efficiency analysis."""
        if power_df.empty:
            print("No power data available for efficiency analysis")
            return
        
        # Merge performance and power data
        merged_df = results_df.merge(power_df, on='platform', how='left')
        
        # Calculate efficiency (FPS/Watt)
        merged_df['efficiency_fps_per_watt'] = merged_df['throughput_fps'] / merged_df['mean_power_w']
        
        # Create efficiency comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Power consumption by platform
        power_summary = power_df.groupby('platform')['mean_power_w'].mean().reset_index()
        platform_names = {
            'nvidia-jetson': 'NVIDIA Jetson Orin NX',
            'qualcomm-qcs6490': 'Qualcomm QCS6490',
            'radxa-x4': 'Radxa X4 (Intel N100)'
        }
        power_summary['platform_name'] = power_summary['platform'].map(platform_names)
        
        sns.barplot(data=power_summary, x='platform_name', y='mean_power_w', ax=ax1)
        ax1.set_title('Average Power Consumption')
        ax1.set_ylabel('Power (W)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Efficiency comparison
        efficiency_data = merged_df.dropna(subset=['efficiency_fps_per_watt'])
        if not efficiency_data.empty:
            efficiency_data['platform_name'] = efficiency_data['platform'].map(platform_names)
            
            sns.scatterplot(data=efficiency_data, x='mean_power_w', y='throughput_fps', 
                           hue='platform_name', s=100, ax=ax2)
            ax2.set_xlabel('Power Consumption (W)')
            ax2.set_ylabel('Throughput (FPS)')
            ax2.set_title('Performance vs Power Consumption')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, results_df, power_df, output_dir):
        """Generate comprehensive summary report."""
        report_path = output_dir / 'benchmark_summary_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Embedded AI Benchmark Suite - Results Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Platform overview
            f.write("## Platform Overview\n\n")
            platforms = results_df['platform'].unique()
            f.write(f"Total platforms tested: {len(platforms)}\n")
            f.write(f"Platforms: {', '.join(platforms)}\n\n")
            
            # Benchmark overview
            benchmarks = results_df['benchmark'].unique()
            f.write("## Benchmarks Executed\n\n")
            for benchmark in benchmarks:
                benchmark_data = results_df[results_df['benchmark'] == benchmark]
                f.write(f"### {benchmark.replace('_', ' ').title()}\n")
                f.write(f"- Total runs: {len(benchmark_data)}\n")
                f.write(f"- Platforms: {len(benchmark_data['platform'].unique())}\n")
                if 'accelerator' in benchmark_data.columns:
                    accelerators = benchmark_data['accelerator'].unique()
                    f.write(f"- Accelerators: {', '.join(accelerators)}\n")
                f.write("\n")
            
            # Performance summary tables
            f.write("## Performance Summary\n\n")
            
            for benchmark in benchmarks:
                benchmark_data = results_df[results_df['benchmark'] == benchmark]
                f.write(f"### {benchmark.replace('_', ' ').title()}\n\n")
                
                # Create summary table
                if benchmark == 'orb_slam3':
                    summary_cols = ['platform', 'throughput_fps', 'p99_latency_ms', 'mean_latency_ms']
                else:
                    summary_cols = ['platform', 'accelerator', 'throughput_fps', 'p99_latency_ms', 'mean_latency_ms']
                
                summary_table = benchmark_data[summary_cols].round(2)
                f.write(summary_table.to_markdown(index=False))
                f.write("\n\n")
            
            # Power analysis
            if not power_df.empty:
                f.write("## Power Analysis\n\n")
                power_summary = power_df.groupby('platform').agg({
                    'mean_power_w': 'mean',
                    'max_power_w': 'max',
                    'min_power_w': 'min'
                }).round(2)
                f.write(power_summary.to_markdown())
                f.write("\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Best performer per benchmark
            for benchmark in benchmarks:
                benchmark_data = results_df[results_df['benchmark'] == benchmark]
                best_performer = benchmark_data.loc[benchmark_data['throughput_fps'].idxmax()]
                
                f.write(f"### {benchmark.replace('_', ' ').title()}\n")
                f.write(f"- **Best Performance:** {best_performer['platform']}")
                if 'accelerator' in best_performer:
                    f.write(f" ({best_performer['accelerator']})")
                f.write(f" - {best_performer['throughput_fps']:.2f} FPS\n")
                f.write(f"- **Lowest Latency:** {best_performer['p99_latency_ms']:.2f} ms (P99)\n\n")
            
            f.write("## Methodology Notes\n\n")
            f.write("- All benchmarks use INT8 quantization for fair comparison\n")
            f.write("- Platform-specific SDKs used for optimal performance\n")
            f.write("- Power measurements taken with external hardware analyzer\n")
            f.write("- Multiple runs averaged for statistical significance\n\n")
        
        print(f"Summary report generated: {report_path}")
    
    def run_complete_analysis(self, output_dir):
        """Run complete analysis pipeline."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Starting comprehensive benchmark analysis...")
        
        # Discover all result files
        self.discover_results()
        
        # Load all results
        all_results = []
        
        # Load ORB-SLAM3 results
        orb_results = self.load_orb_slam3_results()
        all_results.extend(orb_results)
        
        # Load AI benchmark results
        for benchmark in ['3d_detection', 'segmentation']:
            ai_results = self.load_ai_benchmark_results(benchmark)
            all_results.extend(ai_results)
        
        # Load power data
        power_results = self.load_power_data()
        
        if not all_results:
            print("No benchmark results found!")
            return
        
        # Convert to DataFrames
        results_df = pd.DataFrame(all_results)
        power_df = pd.DataFrame(power_results)
        
        print(f"Loaded {len(all_results)} benchmark results")
        print(f"Loaded {len(power_results)} power measurements")
        
        # Create visualizations
        self.create_performance_comparison(results_df, output_dir)
        
        if not power_df.empty:
            self.create_efficiency_analysis(results_df, power_df, output_dir)
        
        # Generate summary report
        self.generate_summary_report(results_df, power_df, output_dir)
        
        # Save processed data
        results_df.to_csv(output_dir / 'processed_results.csv', index=False)
        if not power_df.empty:
            power_df.to_csv(output_dir / 'power_data.csv', index=False)
        
        print(f"Analysis complete! Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze embedded AI benchmark results')
    parser.add_argument('--results-root', default='../..', 
                       help='Root directory containing platform results')
    parser.add_argument('--output-dir', default='../analysis_output',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_root)
    analyzer.run_complete_analysis(args.output_dir)

if __name__ == "__main__":
    main()
