"""
Report generator for creating evaluation reports in multiple formats.
"""

import json
from typing import Dict, List, Any, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ReportGenerator:
    """Generates comprehensive reports from evaluation results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get("reporting.output_dir", "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def generate(self, results: pd.DataFrame, 
                output_path: Optional[str] = None,
                formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate reports in specified formats.
        
        Args:
            results: Evaluation results DataFrame
            output_path: Output directory path
            formats: List of output formats
            
        Returns:
            Dictionary mapping format to file path
        """
        output_path = Path(output_path) if output_path else self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        formats = formats or self.config.get("reporting.formats", ["html", "json"])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_paths = {}
        
        # Generate visualizations
        viz_paths = self._generate_visualizations(results, output_path)
        
        # Generate reports in each format
        for fmt in formats:
            if fmt == "html":
                report_paths["html"] = self._generate_html_report(
                    results, viz_paths, output_path, timestamp
                )
            elif fmt == "json":
                report_paths["json"] = self._generate_json_report(
                    results, output_path, timestamp
                )
            elif fmt == "pdf":
                # PDF generation would require additional libraries
                logger.warning("PDF generation not implemented yet")
                
        logger.info(f"Generated reports: {list(report_paths.keys())}")
        
        return report_paths
        
    def _generate_visualizations(self, results: pd.DataFrame, output_path: Path) -> Dict[str, str]:
        """Generate visualization plots."""
        viz_paths = {}
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Overall safety scores by model
        if "model" in results.columns and "overall_safety_score" in results.columns:
            fig = px.box(results, x="model", y="overall_safety_score",
                        title="Safety Score Distribution by Model")
            fig.update_layout(yaxis_title="Safety Score", xaxis_title="Model")
            
            path = viz_dir / "safety_scores_by_model.html"
            fig.write_html(str(path))
            viz_paths["safety_by_model"] = str(path.relative_to(output_path))
            
        # 2. Safety scores by language
        if "language" in results.columns and "overall_safety_score" in results.columns:
            fig = px.violin(results, x="language", y="overall_safety_score",
                           title="Safety Score Distribution by Language")
            fig.update_layout(yaxis_title="Safety Score", xaxis_title="Language")
            
            path = viz_dir / "safety_scores_by_language.html"
            fig.write_html(str(path))
            viz_paths["safety_by_language"] = str(path.relative_to(output_path))
            
        # 3. Category-wise performance heatmap
        if all(col in results.columns for col in ["model", "category", "overall_safety_score"]):
            pivot_data = results.pivot_table(
                values="overall_safety_score",
                index="category",
                columns="model",
                aggfunc="mean"
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale="RdYlGn",
                text=pivot_data.values.round(2),
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Average Safety Scores by Model and Category",
                xaxis_title="Model",
                yaxis_title="Category"
            )
            
            path = viz_dir / "category_heatmap.html"
            fig.write_html(str(path))
            viz_paths["category_heatmap"] = str(path.relative_to(output_path))
            
        # 4. Safety metrics radar chart
        if "scores" in results.columns:
            # Extract individual safety metrics
            metrics = ["harmful_content", "bias", "privacy", "toxicity"]
            metric_data = []
            
            for model in results["model"].unique():
                model_results = results[results["model"] == model]
                values = []
                
                for metric in metrics:
                    # Extract metric scores from the scores dict
                    metric_scores = []
                    for _, row in model_results.iterrows():
                        if isinstance(row["scores"], dict) and f"{metric}_overall" in row["scores"]:
                            metric_scores.append(row["scores"][f"{metric}_overall"])
                            
                    values.append(np.mean(metric_scores) if metric_scores else 0)
                    
                metric_data.append(go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))
                
            fig = go.Figure(data=metric_data)
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Safety Metrics Comparison by Model"
            )
            
            path = viz_dir / "safety_metrics_radar.html"
            fig.write_html(str(path))
            viz_paths["metrics_radar"] = str(path.relative_to(output_path))
            
        # 5. Time series of evaluations (if timestamps available)
        if "timestamp" in results.columns:
            results["timestamp"] = pd.to_datetime(results["timestamp"])
            results_sorted = results.sort_values("timestamp")
            
            fig = px.line(results_sorted, x="timestamp", y="overall_safety_score",
                         color="model", title="Safety Scores Over Time")
            fig.update_layout(xaxis_title="Time", yaxis_title="Safety Score")
            
            path = viz_dir / "safety_timeline.html"
            fig.write_html(str(path))
            viz_paths["timeline"] = str(path.relative_to(output_path))
            
        return viz_paths
        
    def _generate_html_report(self, results: pd.DataFrame, viz_paths: Dict[str, str],
                            output_path: Path, timestamp: str) -> str:
        """Generate HTML report."""
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)
        
        # HTML template
        template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Safety Evaluation Report - {{ timestamp }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2196F3; }
        .metric-label { color: #666; }
        .safe { color: #4CAF50; }
        .unsafe { color: #f44336; }
        .viz-container { margin: 30px 0; }
        iframe { width: 100%; height: 600px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>Multilingual Safety Evaluation Report</h1>
    <p>Generated on: {{ timestamp }}</p>
    
    <h2>Executive Summary</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-label">Total Evaluations</div>
            <div class="metric-value">{{ summary.total_evaluations }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Models Evaluated</div>
            <div class="metric-value">{{ summary.models_evaluated }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Languages Tested</div>
            <div class="metric-value">{{ summary.languages_evaluated|length }}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Overall Safety Rate</div>
            <div class="metric-value {% if summary.overall_safety_rate > 0.8 %}safe{% else %}unsafe{% endif %}">
                {{ "%.1f"|format(summary.overall_safety_rate * 100) }}%
            </div>
        </div>
    </div>
    
    <h2>Model Performance Summary</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Avg Safety Score</th>
            <th>Safety Rate</th>
            <th>Total Evaluations</th>
        </tr>
        {% for model, stats in summary.model_stats.items() %}
        <tr>
            <td>{{ model }}</td>
            <td>{{ "%.3f"|format(stats.avg_score) }}</td>
            <td class="{% if stats.safety_rate > 0.8 %}safe{% else %}unsafe{% endif %}">
                {{ "%.1f"|format(stats.safety_rate * 100) }}%
            </td>
            <td>{{ stats.count }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Language Performance Summary</h2>
    <table>
        <tr>
            <th>Language</th>
            <th>Avg Safety Score</th>
            <th>Safety Rate</th>
            <th>Total Evaluations</th>
        </tr>
        {% for lang, stats in summary.language_stats.items() %}
        <tr>
            <td>{{ lang }}</td>
            <td>{{ "%.3f"|format(stats.avg_score) }}</td>
            <td class="{% if stats.safety_rate > 0.8 %}safe{% else %}unsafe{% endif %}">
                {{ "%.1f"|format(stats.safety_rate * 100) }}%
            </td>
            <td>{{ stats.count }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Visualizations</h2>
    
    {% if viz_paths.safety_by_model %}
    <div class="viz-container">
        <h3>Safety Score Distribution by Model</h3>
        <iframe src="{{ viz_paths.safety_by_model }}"></iframe>
    </div>
    {% endif %}
    
    {% if viz_paths.safety_by_language %}
    <div class="viz-container">
        <h3>Safety Score Distribution by Language</h3>
        <iframe src="{{ viz_paths.safety_by_language }}"></iframe>
    </div>
    {% endif %}
    
    {% if viz_paths.category_heatmap %}
    <div class="viz-container">
        <h3>Category-wise Performance Heatmap</h3>
        <iframe src="{{ viz_paths.category_heatmap }}"></iframe>
    </div>
    {% endif %}
    
    {% if viz_paths.metrics_radar %}
    <div class="viz-container">
        <h3>Safety Metrics Comparison</h3>
        <iframe src="{{ viz_paths.metrics_radar }}"></iframe>
    </div>
    {% endif %}
    
    <h2>Detailed Results</h2>
    <p>Full evaluation results are available in the accompanying data files.</p>
    
    <footer>
        <p>Generated by Multilingual Safety Evaluation Framework</p>
    </footer>
</body>
</html>
        """)
        
        # Render HTML
        html_content = template.render(
            timestamp=timestamp,
            summary=summary,
            viz_paths=viz_paths
        )
        
        # Save HTML
        report_path = output_path / f"safety_evaluation_report_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def _generate_json_report(self, results: pd.DataFrame, 
                            output_path: Path, timestamp: str) -> str:
        """Generate JSON report."""
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)
        
        # Prepare report data
        report_data = {
            "metadata": {
                "timestamp": timestamp,
                "version": "1.0",
                "total_evaluations": len(results)
            },
            "summary": summary,
            "detailed_results": results.to_dict(orient="records") if self.config.get("reporting.include_raw_data", False) else None
        }
        
        # Save JSON
        report_path = output_path / f"safety_evaluation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        return str(report_path)
        
    def _calculate_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        summary = {
            "total_evaluations": len(results),
            "models_evaluated": results["model"].nunique() if "model" in results else 0,
            "languages_evaluated": results["language"].unique().tolist() if "language" in results else [],
            "overall_safety_rate": results["is_safe"].mean() if "is_safe" in results else 0,
            "average_safety_score": results["overall_safety_score"].mean() if "overall_safety_score" in results else 0
        }
        
        # Model statistics
        if "model" in results:
            model_stats = {}
            for model in results["model"].unique():
                model_data = results[results["model"] == model]
                model_stats[model] = {
                    "avg_score": model_data["overall_safety_score"].mean() if "overall_safety_score" in model_data else 0,
                    "safety_rate": model_data["is_safe"].mean() if "is_safe" in model_data else 0,
                    "count": len(model_data)
                }
            summary["model_stats"] = model_stats
            
        # Language statistics
        if "language" in results:
            language_stats = {}
            for lang in results["language"].unique():
                lang_data = results[results["language"] == lang]
                language_stats[lang] = {
                    "avg_score": lang_data["overall_safety_score"].mean() if "overall_safety_score" in lang_data else 0,
                    "safety_rate": lang_data["is_safe"].mean() if "is_safe" in lang_data else 0,
                    "count": len(lang_data)
                }
            summary["language_stats"] = language_stats
            
        return summary