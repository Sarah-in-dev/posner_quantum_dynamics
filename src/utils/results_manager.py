# src/utils/results_manager.py
"""
Centralized results management for all models
Handles saving, loading, comparison, and database storage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import pickle
import sqlite3
import hashlib
from dataclasses import asdict

class ResultsManager:
    """Manages simulation results with automatic organization and tracking"""
    
    def __init__(self, base_dir: str = "results", experiment_name: str = "default"):
        """
        Initialize results manager
        
        Args:
            base_dir: Base directory for all results
            experiment_name: Name of current experiment
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create directory structure
        self.experiment_dir = self.base_dir / experiment_name
        self.run_dir = self.experiment_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.data_dir = self.run_dir / "data"
        self.figures_dir = self.run_dir / "figures"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        
        for dir in [self.data_dir, self.figures_dir, self.checkpoints_dir]:
            dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.experiment_dir / "results.db"
        self.init_database()
        
        # Track current results
        self.current_results = []
        
        print(f"ResultsManager initialized: {self.run_dir}")
    
    def init_database(self):
        """Initialize SQLite database for results tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                experiment_name TEXT,
                timestamp DATETIME,
                tag TEXT,
                model_version TEXT,
                
                -- Key parameters
                n_channels INTEGER,
                channel_open_rate REAL,
                channel_close_rate REAL,
                grid_size INTEGER,
                dt REAL,
                duration REAL,
                
                -- Key metrics
                peak_posner_nm REAL,
                mean_posner_nm REAL,
                coherence_time_s REAL,
                spatial_heterogeneity REAL,
                hotspot_lifetime_s REAL,
                channel_open_fraction REAL,
                mean_burst_duration_s REAL,
                
                -- File paths
                data_path TEXT,
                figure_path TEXT,
                
                -- Metadata
                parameter_hash TEXT,
                notes TEXT
            )
        """)
        
        # Create parameter sets table (for tracking unique configurations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameter_sets (
                hash TEXT PRIMARY KEY,
                parameters TEXT,
                first_run_id TEXT,
                n_runs INTEGER DEFAULT 1
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON simulations(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON simulations(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_peak_posner ON simulations(peak_posner_nm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiment ON simulations(experiment_name)")
        
        conn.commit()
        conn.close()
    
    def save_results(self, results: Any, tag: str = "default", notes: str = "") -> Path:
        """
        Save simulation results with automatic organization
        
        Args:
            results: SimulationResults object
            tag: Descriptive tag for this run
            notes: Optional notes about the run
        
        Returns:
            Path to saved results
        """
        # Generate filename
        filename = f"{self.experiment_name}_{self.run_id}_{tag}"
        filepath = self.data_dir / filename
        
        # Save results files
        results.save(filepath)
        
        # Add to current results
        self.current_results.append({
            'tag': tag,
            'results': results,
            'filepath': filepath
        })
        
        # Save to database
        self.save_to_database(results, tag, filepath, notes)
        
        # Create "latest" symlink for easy access
        latest_path = self.experiment_dir / f"latest_{tag}"
        if latest_path.exists():
            latest_path.unlink()
        
        # Create relative symlink
        try:
            relative_path = Path("..") / ".." / filepath.relative_to(self.base_dir)
            latest_path.symlink_to(relative_path)
        except:
            # Fallback if symlink fails (e.g., on Windows)
            pass
        
        return filepath
    
    def save_to_database(self, results: Any, tag: str, filepath: Path, notes: str = ""):
        """Save results metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate parameter hash
        param_str = json.dumps(results.parameters, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        # Extract key parameters (with defaults for missing)
        params = results.parameters
        
        # Insert into simulations table
        cursor.execute("""
            INSERT INTO simulations (
                run_id, experiment_name, timestamp, tag, model_version,
                n_channels, channel_open_rate, channel_close_rate, 
                grid_size, dt, duration,
                peak_posner_nm, mean_posner_nm, coherence_time_s,
                spatial_heterogeneity, hotspot_lifetime_s,
                channel_open_fraction, mean_burst_duration_s,
                data_path, parameter_hash, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.run_id,
            self.experiment_name,
            results.timestamp,
            tag,
            results.model_version,
            params.get('n_channels', 0),
            params.get('channel_open_rate', 0),
            params.get('channel_close_rate', 0),
            params.get('grid_size', 0),
            params.get('dt', 0),
            params.get('duration', 0),
            results.peak_posner,
            results.mean_posner,
            results.coherence_time,
            results.spatial_heterogeneity,
            results.hotspot_lifetime,
            results.channel_open_fraction,
            results.mean_burst_duration,
            str(filepath),
            param_hash,
            notes
        ))
        
        # Update parameter sets table
        cursor.execute("""
            INSERT OR REPLACE INTO parameter_sets (hash, parameters, first_run_id, n_runs)
            VALUES (
                ?,
                ?,
                COALESCE((SELECT first_run_id FROM parameter_sets WHERE hash = ?), ?),
                COALESCE((SELECT n_runs + 1 FROM parameter_sets WHERE hash = ?), 1)
            )
        """, (param_hash, param_str, param_hash, self.run_id, param_hash))
        
        conn.commit()
        conn.close()
    
    def save_checkpoint(self, results_list: List = None, tag: str = "checkpoint"):
        """Save intermediate results during long runs"""
        if results_list is None:
            results_list = self.current_results
        
        checkpoint_file = self.checkpoints_dir / f"{tag}_{datetime.now().strftime('%H%M%S')}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(results_list, f)
        
        print(f"Checkpoint saved: {checkpoint_file}")
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: Path = None) -> List:
        """Load results from checkpoint"""
        if checkpoint_file is None:
            # Get latest checkpoint
            checkpoints = list(self.checkpoints_dir.glob("*.pkl"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_file = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        with open(checkpoint_file, 'rb') as f:
            results_list = pickle.load(f)
        
        print(f"Loaded {len(results_list)} results from {checkpoint_file}")
        return results_list
    
    def create_comparison_dataframe(self, results_list: List = None) -> pd.DataFrame:
        """Create DataFrame for comparing multiple runs"""
        if results_list is None:
            results_list = self.current_results
        
        data = []
        for item in results_list:
            if isinstance(item, dict):
                results = item['results']
                tag = item.get('tag', 'unknown')
            else:
                results = item
                tag = 'unknown'
            
            row = {
                'tag': tag,
                'timestamp': results.timestamp,
                'model_version': results.model_version,
                'peak_posner_nm': results.peak_posner,
                'mean_posner_nm': results.mean_posner,
                'coherence_time_s': results.coherence_time,
                'spatial_heterogeneity': results.spatial_heterogeneity,
                'hotspot_lifetime_s': results.hotspot_lifetime,
                'channel_open_fraction': results.channel_open_fraction,
                'mean_burst_duration_s': results.mean_burst_duration
            }
            
            # Add parameters
            for key, value in results.parameters.items():
                if isinstance(value, (int, float, str, bool)):
                    row[f'param_{key}'] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = self.run_dir / "comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"Comparison saved to {csv_path}")
        
        return df
    
    def query_database(self, query: str = None, **kwargs) -> pd.DataFrame:
        """
        Query the results database
        
        Args:
            query: SQL query string (if provided, kwargs ignored)
            **kwargs: Column filters (e.g., model_version='4.0')
        
        Returns:
            DataFrame with query results
        """
        conn = sqlite3.connect(self.db_path)
        
        if query is None:
            # Build query from kwargs
            conditions = []
            values = []
            
            for key, value in kwargs.items():
                conditions.append(f"{key} = ?")
                values.append(value)
            
            if conditions:
                where_clause = " WHERE " + " AND ".join(conditions)
            else:
                where_clause = ""
            
            query = f"SELECT * FROM simulations{where_clause} ORDER BY timestamp DESC"
            df = pd.read_sql_query(query, conn, params=values)
        else:
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df
    
    def get_best_results(self, metric: str = "peak_posner_nm", 
                        n: int = 10) -> pd.DataFrame:
        """Get top N results by a specific metric"""
        query = f"""
            SELECT * FROM simulations 
            ORDER BY {metric} DESC 
            LIMIT {n}
        """
        return self.query_database(query)
    
    def get_parameter_sensitivity(self, param_name: str, 
                                 metric: str = "peak_posner_nm") -> pd.DataFrame:
        """Analyze sensitivity of a metric to a parameter"""
        query = f"""
            SELECT {param_name}, 
                   AVG({metric}) as mean_{metric},
                   STDEV({metric}) as std_{metric},
                   COUNT(*) as n_runs
            FROM simulations
            WHERE {param_name} IS NOT NULL
            GROUP BY {param_name}
            ORDER BY {param_name}
        """
        return self.query_database(query)
    
    def generate_report(self) -> str:
        """Generate summary report of current run"""
        report_lines = [
            f"# Experiment Report: {self.experiment_name}",
            f"## Run ID: {self.run_id}",
            f"## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"### Summary Statistics",
            f"- Total simulations: {len(self.current_results)}",
            ""
        ]
        
        if self.current_results:
            df = self.create_comparison_dataframe()
            
            # Add summary statistics
            report_lines.append("### Performance Metrics")
            for col in ['peak_posner_nm', 'coherence_time_s', 'spatial_heterogeneity']:
                if col in df.columns:
                    report_lines.append(
                        f"- {col}: {df[col].mean():.3f} ± {df[col].std():.3f}"
                    )
            
            report_lines.append("")
            report_lines.append("### Best Results")
            if 'peak_posner_nm' in df.columns:
                best_idx = df['peak_posner_nm'].idxmax()
                best_row = df.loc[best_idx]
                report_lines.append(f"- Best peak Posner: {best_row['peak_posner_nm']:.2f} nM")
                report_lines.append(f"  - Tag: {best_row['tag']}")
                report_lines.append(f"  - Coherence time: {best_row.get('coherence_time_s', 0):.3f} s")
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.run_dir / "report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {report_path}")
        return report

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_summary_figure(results: Any, save_path: Path = None):
    """Create standard summary figure for results"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Posner evolution
    ax1 = fig.add_subplot(gs[0, :])
    total_posner = np.sum(results.posner_map, axis=(1, 2)) * 1e9  # nM
    ax1.plot(results.time, total_posner)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Total Posner (nM)')
    ax1.set_title('Posner Evolution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Final spatial map
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(results.posner_map[-1] * 1e9, cmap='hot', 
                    extent=[-200, 200, -200, 200])
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    ax2.set_title('Final Posner Distribution')
    plt.colorbar(im, ax=ax2, label='[Posner] (nM)')
    
    # 3. Channel states
    ax3 = fig.add_subplot(gs[1, 1])
    channel_raster = results.channel_states.T
    ax3.imshow(channel_raster, aspect='auto', cmap='RdYlGn',
              extent=[0, results.time[-1], 0, channel_raster.shape[0]])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Channel #')
    ax3.set_title('Channel States (0=closed, 1=open, 2=inactive)')
    
    # 4. Calcium map at peak
    ax4 = fig.add_subplot(gs[1, 2])
    peak_idx = np.argmax(np.sum(results.posner_map, axis=(1, 2)))
    im = ax4.imshow(results.calcium_map[peak_idx] * 1e6, cmap='viridis',
                   extent=[-200, 200, -200, 200])
    ax4.set_xlabel('x (nm)')
    ax4.set_ylabel('y (nm)')
    ax4.set_title('Calcium at Peak Posner')
    plt.colorbar(im, ax=ax4, label='[Ca²⁺] (μM)')
    
    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    metrics_text = f"""
    Key Metrics:
    • Peak Posner: {results.peak_posner:.1f} nM
    • Mean Posner: {results.mean_posner:.1f} nM
    • Coherence Time: {results.coherence_time:.3f} s
    • Spatial Heterogeneity: {results.spatial_heterogeneity:.3f}
    • Hotspot Lifetime: {results.hotspot_lifetime:.3f} s
    • Channel Open Fraction: {results.channel_open_fraction:.3f}
    • Mean Burst Duration: {results.mean_burst_duration:.3f} s
    """
    
    ax5.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
            family='monospace')
    
    plt.suptitle(f"Model {results.model_version} - {results.timestamp}", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

def create_comparison_plots(results_list: List, save_path: Path = None):
    """Create comparison plots for multiple results"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Extract metrics
    tags = []
    peak_posners = []
    coherence_times = []
    heterogeneities = []
    open_fractions = []
    
    for item in results_list:
        if isinstance(item, dict):
            results = item['results']
            tag = item.get('tag', 'unknown')
        else:
            results = item
            tag = 'unknown'
        
        tags.append(tag)
        peak_posners.append(results.peak_posner)
        coherence_times.append(results.coherence_time)
        heterogeneities.append(results.spatial_heterogeneity)
        open_fractions.append(results.channel_open_fraction)
    
    # Plot comparisons
    x = np.arange(len(tags))
    
    axes[0, 0].bar(x, peak_posners)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(tags, rotation=45)
    axes[0, 0].set_ylabel('Peak Posner (nM)')
    axes[0, 0].set_title('Peak Posner Comparison')
    
    axes[0, 1].bar(x, coherence_times)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(tags, rotation=45)
    axes[0, 1].set_ylabel('Coherence Time (s)')
    axes[0, 1].set_title('Coherence Time Comparison')
    
    axes[0, 2].bar(x, heterogeneities)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(tags, rotation=45)
    axes[0, 2].set_ylabel('Spatial Heterogeneity')
    axes[0, 2].set_title('Spatial Heterogeneity')
    
    axes[1, 0].scatter(open_fractions, peak_posners)
    axes[1, 0].set_xlabel('Channel Open Fraction')
    axes[1, 0].set_ylabel('Peak Posner (nM)')
    axes[1, 0].set_title('Open Fraction vs Peak Posner')
    
    axes[1, 1].scatter(peak_posners, coherence_times)
    axes[1, 1].set_xlabel('Peak Posner (nM)')
    axes[1, 1].set_ylabel('Coherence Time (s)')
    axes[1, 1].set_title('Posner vs Coherence Time')
    
    # Parameter correlation matrix
    if len(results_list) > 1:
        import pandas as pd
        df = pd.DataFrame({
            'Peak Posner': peak_posners,
            'Coherence': coherence_times,
            'Heterogeneity': heterogeneities,
            'Open Fraction': open_fractions
        })
        
        corr = df.corr()
        im = axes[1, 2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_xticks(range(len(corr.columns)))
        axes[1, 2].set_yticks(range(len(corr.columns)))
        axes[1, 2].set_xticklabels(corr.columns, rotation=45)
        axes[1, 2].set_yticklabels(corr.columns)
        axes[1, 2].set_title('Correlation Matrix')
        
        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                axes[1, 2].text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha='center', va='center')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")
    
    return fig