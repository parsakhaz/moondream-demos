"""
Moondream Benchmark Visualization Script
--------------------------------------
Creates visualizations from stored benchmark results to analyze:
- Performance trends over time
- Comparisons across different models
- Memory usage patterns
- Runtime environment impacts
"""

# ===== STEP 1: Install required packages =====
!pip install -q pandas matplotlib seaborn plotly

# ===== STEP 2: Import libraries =====
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from google.colab import drive

# ===== STEP 3: Mount Google Drive =====
drive.mount('/content/drive')

def load_benchmark_data():
    """Load all benchmark results into a pandas DataFrame"""
    benchmark_dir = '/content/drive/MyDrive/md-benchmarks'
    if not os.path.exists(benchmark_dir):
        raise Exception("No benchmark directory found!")
    
    data = []
    for filename in os.listdir(benchmark_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(benchmark_dir, filename)
        with open(filepath, 'r') as f:
            benchmark = json.load(f)
            
            # Extract metadata
            timestamp = datetime.fromisoformat(benchmark['metadata']['timestamp'])
            system_info = benchmark['metadata']['system_info']
            
            # Process each model's results
            for model_name, model_data in benchmark['results'].items():
                # Process each run for this model
                for run in model_data['runs']:
                    row = {
                        'timestamp': timestamp,
                        'model': model_name,
                        # System info
                        'memory_gb': system_info['memory_gb'],
                        'cpu_count': system_info['cpu_count'],
                        'platform': system_info['platform'],
                        'gpu_name': system_info['gpu_name'],
                        'has_gpu': system_info['gpu_available'],
                        # Run metrics
                        'initial_memory': run['initial_memory'],
                        'model_load_time': run['model_load_time'],
                        'memory_increase': run['memory_increase'],
                        'cold_encode_time': run['cold_encode_time'],
                        'avg_warm_encode_time': run['avg_warm_encode_time'],
                        'ttft': run['ttft'],
                        'caption_time': run['caption_time'],
                        'tokens_per_second': run['tokens_per_second'],
                        'num_tokens': run['num_tokens']
                    }
                    data.append(row)
    
    return pd.DataFrame(data)

def plot_performance_over_time(df):
    """Create interactive time series plots for key metrics"""
    metrics = ['ttft', 'model_load_time', 'cold_encode_time', 'avg_warm_encode_time',
              'tokens_per_second', 'memory_increase']
    titles = ['Time to First Token', 'Model Load Time', 'Cold Encoding Time', 
              'Warm Encoding Time', 'Generation Speed', 'Memory Usage']
    
    for metric, title in zip(metrics, titles):
        fig = px.line(df, x='timestamp', y=metric, color='model',
                     title=f'{title} Over Time',
                     labels={'timestamp': 'Benchmark Date',
                            metric: title})
        fig.show()

def plot_model_comparisons(df):
    """Create box plots comparing model performances"""
    metrics = ['ttft', 'model_load_time', 'cold_encode_time', 'avg_warm_encode_time',
              'tokens_per_second', 'memory_increase']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Model Performance Comparisons', fontsize=16)
    
    for ax, metric in zip(axes.flat, metrics):
        sns.boxplot(data=df, x='model', y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(f'{metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.show()

def plot_memory_analysis(df):
    """Analyze and visualize memory usage patterns"""
    fig = px.scatter(df, x='memory_gb', y='memory_increase',
                    color='model', size='model_load_time',
                    title='Memory Usage vs System Memory',
                    labels={'memory_gb': 'System Memory (GB)',
                           'memory_increase': 'Memory Usage (MB)'})
    fig.show()

def plot_performance_heatmap(df):
    """Create a heatmap showing correlations between metrics"""
    metrics = ['ttft', 'model_load_time', 'cold_encode_time', 'avg_warm_encode_time',
              'tokens_per_second', 'memory_increase', 'memory_gb', 
              'cpu_count', 'initial_memory']
    
    corr = df[metrics].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Metric Correlations')
    plt.show()

def plot_hardware_performance_comparison(df):
    """Compare performance across different hardware configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hardware Performance Analysis', fontsize=16)
    
    # GPU vs CPU comparison
    ax = axes[0, 0]
    sns.boxplot(data=df, x='has_gpu', y='ttft', hue='model', ax=ax)
    ax.set_title('Time to First Token: GPU vs CPU')
    
    # Performance by GPU model
    ax = axes[0, 1]
    gpu_data = df[df['has_gpu']].dropna(subset=['gpu_name'])
    if not gpu_data.empty:
        sns.boxplot(data=gpu_data, x='gpu_name', y='tokens_per_second', hue='model', ax=ax)
        ax.set_title('Generation Speed by GPU Model')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    else:
        ax.text(0.5, 0.5, 'No GPU data available', ha='center', va='center')
    
    # Memory usage by platform
    ax = axes[1, 0]
    sns.boxplot(data=df, x='platform', y='memory_increase', hue='model', ax=ax)
    ax.set_title('Memory Usage by Platform')
    
    # CPU count impact
    ax = axes[1, 1]
    sns.scatterplot(data=df, x='cpu_count', y='model_load_time', 
                    hue='model', size='memory_increase', ax=ax)
    ax.set_title('Load Time vs CPU Count')
    
    plt.tight_layout()
    plt.show()

def generate_summary_stats(df):
    """Generate and display summary statistics"""
    print("\n📊 Summary Statistics by Model")
    print("=" * 80)
    
    metrics = ['ttft', 'model_load_time', 'cold_encode_time', 'avg_warm_encode_time',
              'tokens_per_second', 'memory_increase']
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n🤖 {model}")
        for metric in metrics:
            stats = model_data[metric].describe()
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
            print(f"  Std:  {stats['std']:.2f}")

def generate_hardware_summary(df):
    """Generate summary of performance across different hardware configurations"""
    print("\n🖥️ Hardware Configuration Summary")
    print("=" * 80)
    
    # Group by hardware configuration
    configs = df.groupby(['platform', 'gpu_name']).agg({
        'ttft': ['mean', 'std'],
        'cold_encode_time': ['mean', 'std'],
        'avg_warm_encode_time': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
        'model_load_time': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance by Hardware Configuration:")
    print(configs)

def generate_encoding_summary(df):
    """Generate summary of encoding performance"""
    print("\n🖼️ Image Encoding Performance Summary")
    print("=" * 80)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n📸 {model}")
        
        metrics = {
            'Cold Encode Time': 'cold_encode_time',
            'Warm Encode Time': 'avg_warm_encode_time'
        }
        
        for label, metric in metrics.items():
            stats = model_data[metric].describe()
            print(f"\n{label}:")
            print(f"  Mean: {stats['mean']:.3f}s")
            print(f"  Min:  {stats['min']:.3f}s")
            print(f"  Max:  {stats['max']:.3f}s")
            print(f"  Std:  {stats['std']:.3f}s")

def main():
    """Main function to generate all visualizations"""
    print("🎨 Loading benchmark data...")
    try:
        df = load_benchmark_data()
        
        print("\n📈 Generating visualizations...")
        
        # Time series analysis
        print("\nPlotting performance over time...")
        plot_performance_over_time(df)
        
        # Model comparisons
        print("\nCreating model comparison plots...")
        plot_model_comparisons(df)
        
        # Memory analysis
        print("\nAnalyzing memory patterns...")
        plot_memory_analysis(df)
        
        # Correlation heatmap
        print("\nGenerating correlation heatmap...")
        plot_performance_heatmap(df)
        
        # Summary statistics
        print("\nCalculating summary statistics...")
        generate_summary_stats(df)
        
        # Hardware-specific analysis
        print("\nAnalyzing hardware performance patterns...")
        plot_hardware_performance_comparison(df)
        generate_hardware_summary(df)
        
        # Encoding performance analysis
        print("\nAnalyzing encoding performance...")
        generate_encoding_summary(df)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()