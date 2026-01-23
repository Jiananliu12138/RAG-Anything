"""
评估结果可视化工具
生成评估结果的可视化图表（可选功能）
"""

import json
from typing import Dict, Any, List
from pathlib import Path


class EvaluationVisualizer:
    """评估结果可视化类"""
    
    def __init__(self, results_path: str):
        """
        初始化可视化器
        
        Args:
            results_path: 评估结果 JSON 文件路径
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def plot_retrieval_metrics(self, output_path: str = "./retrieval_metrics.png"):
        """
        绘制检索指标对比图
        
        Args:
            output_path: 输出图片路径
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 提取检索指标
            retriever_summary = self.results.get("component_level", {}).get("retriever", {}).get("summary", {})
            
            # 按 K 值分组
            metrics_by_k = {}
            for metric_name, value in retriever_summary.items():
                if "@" in metric_name and "_mean" in metric_name:
                    # 提取指标类型和 K 值
                    parts = metric_name.split("@")
                    metric_type = parts[0]
                    k_and_stat = parts[1].split("_")
                    k = int(k_and_stat[0])
                    
                    if k not in metrics_by_k:
                        metrics_by_k[k] = {}
                    metrics_by_k[k][metric_type] = value
            
            # 准备数据
            k_values = sorted(metrics_by_k.keys())
            metric_types = ["Precision", "Recall", "HitRate", "NDCG"]
            
            # 绘图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for metric_type in metric_types:
                values = [metrics_by_k[k].get(metric_type, 0) for k in k_values]
                ax.plot(k_values, values, marker='o', label=metric_type, linewidth=2)
            
            ax.set_xlabel('K', fontsize=12)
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Retrieval Metrics @ Different K Values', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 检索指标图已保存: {output_path}")
            
        except ImportError:
            print("⚠️  需要安装 matplotlib: pip install matplotlib")
    
    def plot_generation_metrics_radar(self, output_path: str = "./generation_radar.png"):
        """
        绘制生成指标雷达图
        
        Args:
            output_path: 输出图片路径
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 提取生成指标
            generator_summary = self.results.get("component_level", {}).get("generator", {}).get("summary", {})
            
            # 选择要展示的指标
            metrics_to_plot = {}
            for key, value in generator_summary.items():
                if "mean" in key:
                    metric_name = key.replace("_mean", "")
                    metrics_to_plot[metric_name] = value
            
            if not metrics_to_plot:
                print("⚠️  没有生成指标数据可绘制")
                return
            
            # 准备雷达图数据
            categories = list(metrics_to_plot.keys())
            values = list(metrics_to_plot.values())
            
            # 补全到圆形
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合雷达图
            angles += angles[:1]
            
            # 绘制雷达图
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, color='#00d9ff', label='Scores')
            ax.fill(angles, values, alpha=0.25, color='#00d9ff')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=10)
            ax.set_ylim(0, 1)
            ax.set_title('Generation Metrics Radar Chart', size=14, fontweight='bold', pad=20)
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 生成指标雷达图已保存: {output_path}")
            
        except ImportError:
            print("⚠️  需要安装 matplotlib: pip install matplotlib")
    
    def generate_html_report(self, output_path: str = "./evaluation_report.html"):
        """
        生成交互式 HTML 报告
        
        Args:
            output_path: 输出 HTML 文件路径
        """
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>RAG-Anything 评估报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #00d9ff;
            border-bottom: 3px solid #00d9ff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #00d9ff;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .metric-value {{
            font-weight: bold;
            color: #00d9ff;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>🚀 RAG-Anything 评估报告</h1>
    
    <div class="section">
        <h2>📊 评估概览</h2>
        <p><strong>评估时间:</strong> {eval_time}</p>
        <p><strong>数据集:</strong> {dataset_name}</p>
        <p><strong>查询数量:</strong> {query_count}</p>
    </div>
    
    {component_tables}
    
    {end_to_end_tables}
    
</body>
</html>
"""
        
        # 提取数据
        eval_time = self.results['metadata']['evaluation_time']
        dataset_name = self.results['metadata']['dataset'].get('name', 'N/A')
        query_count = len(self.results['metadata']['dataset'].get('queries', []))
        
        # 生成组件级评估表格
        component_tables = self._generate_component_tables_html()
        
        # 生成端到端评估表格
        end_to_end_tables = self._generate_end_to_end_tables_html()
        
        # 填充模板
        html_content = html_template.format(
            eval_time=eval_time,
            dataset_name=dataset_name,
            query_count=query_count,
            component_tables=component_tables,
            end_to_end_tables=end_to_end_tables
        )
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML 报告已生成: {output_path}")
    
    def _generate_component_tables_html(self) -> str:
        """生成组件级评估的 HTML 表格"""
        html = '<div class="section"><h2>📈 组件级评估</h2>'
        
        for component, data in self.results.get("component_level", {}).items():
            html += f'<h3>{component.upper()}</h3>'
            html += '<table><tr><th>指标</th><th>值</th></tr>'
            for metric, value in data.get("summary", {}).items():
                html += f'<tr><td>{metric}</td><td class="metric-value">{value:.4f}</td></tr>'
            html += '</table>'
        
        html += '</div>'
        return html
    
    def _generate_end_to_end_tables_html(self) -> str:
        """生成端到端评估的 HTML 表格"""
        html = '<div class="section"><h2>🎯 端到端评估</h2>'
        
        for eval_type, data in self.results.get("end_to_end", {}).items():
            html += f'<h3>{eval_type.upper()}</h3>'
            
            if isinstance(data, dict):
                # 处理嵌套结构（如 QA 评估包含 retriever 和 generator）
                for sub_component, sub_data in data.items():
                    if isinstance(sub_data, dict) and "summary" in sub_data:
                        html += f'<h4>{sub_component}</h4>'
                        html += '<table><tr><th>指标</th><th>值</th></tr>'
                        for metric, value in sub_data["summary"].items():
                            html += f'<tr><td>{metric}</td><td class="metric-value">{value:.4f}</td></tr>'
                        html += '</table>'
                
                # 直接有 summary 的情况
                if "summary" in data:
                    html += '<table><tr><th>指标</th><th>值</th></tr>'
                    for metric, value in data["summary"].items():
                        html += f'<tr><td>{metric}</td><td class="metric-value">{value:.4f}</td></tr>'
                    html += '</table>'
        
        html += '</div>'
        return html
