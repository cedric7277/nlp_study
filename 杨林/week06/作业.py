"""
文本分类方法对比分析报告生成器

对比四种方法：
1. BERT微调（无类别权重）
2. BERT微调（有类别权重）
3. LLM Zero-shot
4. LLM SFT微调

生成：
- 性能对比表格
- 可视化图表
- 详细分析报告
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DATA_DIR = ROOT / "data"

def load_results():
    """加载所有训练结果"""
    results = {}
    
    with open(OUTPUTS_DIR / "train_log_cls.json", "r", encoding="utf-8") as f:
        cls_log = json.load(f)
        results["BERT（无权重）"] = {
            "val_acc": cls_log[-1]["val_acc"],
            "val_macro_f1": cls_log[-1]["val_macro_f1"],
            "train_acc": cls_log[-1]["train_acc"],
            "train_loss": cls_log[-1]["train_loss"],
            "epochs": len(cls_log)
        }
    
    with open(OUTPUTS_DIR / "train_log_cls_weighted.json", "r", encoding="utf-8") as f:
        cls_weighted_log = json.load(f)
        results["BERT（有权重）"] = {
            "val_acc": cls_weighted_log[-1]["val_acc"],
            "val_macro_f1": cls_weighted_log[-1]["val_macro_f1"],
            "train_acc": cls_weighted_log[-1]["train_acc"],
            "train_loss": cls_weighted_log[-1]["train_loss"],
            "epochs": len(cls_weighted_log)
        }
    
    with open(OUTPUTS_DIR / "llm_zero_shot_results.json", "r", encoding="utf-8") as f:
        zero_shot = json.load(f)
        results["LLM Zero-shot"] = {
            "val_acc": zero_shot["accuracy"],
            "total": zero_shot["total"],
            "correct": zero_shot["correct"],
            "unparseable": zero_shot["unparseable"]
        }
    
    with open(OUTPUTS_DIR / "llm_sft_results.json", "r", encoding="utf-8") as f:
        sft = json.load(f)
        results["LLM SFT"] = {
            "val_acc": sft["accuracy"],
            "total": sft["total"],
            "correct": sft["correct"],
            "unparseable": sft["unparseable"]
        }
    
    return results

def load_label_distribution():
    """加载标签分布信息"""
    with open(DATA_DIR / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    
    with open(DATA_DIR / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    label_counts = {}
    for item in train_data:
        label = item["label"]
        label_name = label_map["id2name"][str(label)]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    return label_counts, label_map

def plot_performance_comparison(results):
    """绘制性能对比图"""
    methods = list(results.keys())
    accuracies = [results[m]["val_acc"] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax1.bar(range(len(methods)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, fontsize=11, rotation=15, ha='right')
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('不同方法准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    bert_methods = ["BERT（无权重）", "BERT（有权重）"]
    bert_acc = [results[m]["val_acc"] for m in bert_methods]
    bert_f1 = [results[m]["val_macro_f1"] for m in bert_methods]
    
    x = np.arange(len(bert_methods))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, bert_acc, width, label='准确率', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, bert_f1, width, label='Macro F1', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('分数', fontsize=12)
    ax2.set_title('BERT方法详细对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bert_methods, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ 性能对比图已保存: {FIGURES_DIR / 'method_comparison.png'}")
    plt.close()

def plot_llm_comparison(results):
    """绘制LLM方法对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    llm_methods = ["LLM Zero-shot", "LLM SFT"]
    llm_acc = [results[m]["val_acc"] for m in llm_methods]
    
    colors = ['#e74c3c', '#f39c12']
    bars = ax1.bar(range(len(llm_methods)), llm_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(llm_methods)))
    ax1.set_xticklabels(llm_methods, fontsize=11)
    ax1.set_ylabel('准确率', fontsize=12)
    ax1.set_title('LLM方法性能对比', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, acc in zip(bars, llm_acc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2 = axes[1]
    correct = [results[m]["correct"] for m in llm_methods]
    unparseable = [results[m]["unparseable"] for m in llm_methods]
    wrong = [results[m]["total"] - results[m]["correct"] - results[m]["unparseable"] for m in llm_methods]
    
    x = np.arange(len(llm_methods))
    width = 0.6
    
    bars1 = ax2.bar(x, correct, width, label='正确', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x, wrong, width, bottom=correct, label='错误', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax2.bar(x, unparseable, width, bottom=[c+w for c,w in zip(correct, wrong)], 
                    label='无法解析', color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('样本数量', fontsize=12)
    ax2.set_title('LLM预测结果分布', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(llm_methods, fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "llm_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ LLM对比图已保存: {FIGURES_DIR / 'llm_comparison.png'}")
    plt.close()

def plot_training_curves():
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    with open(OUTPUTS_DIR / "train_log_cls.json", "r", encoding="utf-8") as f:
        cls_log = json.load(f)
    
    with open(OUTPUTS_DIR / "train_log_cls_weighted.json", "r", encoding="utf-8") as f:
        cls_weighted_log = json.load(f)
    
    ax1 = axes[0]
    epochs = range(1, len(cls_log) + 1)
    
    ax1.plot(epochs, [e["train_loss"] for e in cls_log], 'o-', label='BERT（无权重）', 
             color='#3498db', linewidth=2, markersize=8)
    ax1.plot(epochs, [e["train_loss"] for e in cls_weighted_log], 's-', label='BERT（有权重）', 
             color='#2ecc71', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('训练损失', fontsize=12)
    ax1.set_title('训练损失曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2 = axes[1]
    ax2.plot(epochs, [e["val_acc"] for e in cls_log], 'o-', label='BERT（无权重）', 
             color='#3498db', linewidth=2, markersize=8)
    ax2.plot(epochs, [e["val_acc"] for e in cls_weighted_log], 's-', label='BERT（有权重）', 
             color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(epochs, [e["train_acc"] for e in cls_log], 'o--', label='BERT（无权重）- 训练', 
             color='#3498db', linewidth=1.5, markersize=6, alpha=0.5)
    ax2.plot(epochs, [e["train_acc"] for e in cls_weighted_log], 's--', label='BERT（有权重）- 训练', 
             color='#2ecc71', linewidth=1.5, markersize=6, alpha=0.5)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.set_title('验证集准确率曲线', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim(0.4, 0.8)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_curves.png", dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {FIGURES_DIR / 'training_curves.png'}")
    plt.close()

def generate_report(results, label_counts):
    """生成详细分析报告"""
    report = []
    report.append("=" * 80)
    report.append("文本分类方法对比分析报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("一、实验概述")
    report.append("-" * 80)
    report.append("本实验对比了四种文本分类方法在中文新闻标题分类任务上的表现：")
    report.append("1. BERT微调（无类别权重）")
    report.append("2. BERT微调（有类别权重）")
    report.append("3. LLM Zero-shot（Qwen2-0.5B-Instruct）")
    report.append("4. LLM SFT微调（Qwen2-0.5B-Instruct）")
    report.append("")
    report.append(f"数据集规模: 训练集样本数 {sum(label_counts.values())} 条")
    report.append(f"分类类别: {len(label_counts)} 个类别")
    report.append("")
    
    report.append("二、性能对比总结")
    report.append("-" * 80)
    report.append(f"{'方法':<20} {'准确率':<15} {'Macro F1':<15} {'备注'}")
    report.append("-" * 80)
    
    for method, data in results.items():
        if "val_macro_f1" in data:
            report.append(f"{method:<20} {data['val_acc']:.2%}         {data['val_macro_f1']:.4f}         BERT微调")
        else:
            unparseable_rate = data['unparseable'] / data['total'] * 100
            report.append(f"{method:<20} {data['val_acc']:.2%}         {'N/A':<15} 无法解析: {unparseable_rate:.1f}%")
    
    report.append("")
    
    report.append("三、详细分析")
    report.append("-" * 80)
    report.append("")
    
    report.append("1. BERT微调方法对比")
    report.append("   - 无权重版本: 验证集准确率 {:.2%}, Macro F1 {:.4f}".format(
        results["BERT（无权重）"]["val_acc"], results["BERT（无权重）"]["val_macro_f1"]))
    report.append("   - 有权重版本: 验证集准确率 {:.2%}, Macro F1 {:.4f}".format(
        results["BERT（有权重）"]["val_acc"], results["BERT（有权重）"]["val_macro_f1"]))
    report.append("   - 结论: 类别权重对准确率略有负面影响，但提升了Macro F1，说明对少数类别的识别更均衡")
    report.append("")
    
    report.append("2. LLM方法对比")
    report.append("   - Zero-shot: 准确率 {:.2%}, 无法解析率 {:.1f}%".format(
        results["LLM Zero-shot"]["val_acc"], 
        results["LLM Zero-shot"]["unparseable"] / results["LLM Zero-shot"]["total"] * 100))
    report.append("   - SFT微调: 准确率 {:.2%}, 无法解析率 {:.1f}%".format(
        results["LLM SFT"]["val_acc"],
        results["LLM SFT"]["unparseable"] / results["LLM SFT"]["total"] * 100))
    report.append("   - 结论: SFT微调显著提升了LLM的分类性能，准确率提升{:.1f}个百分点".format(
        (results["LLM SFT"]["val_acc"] - results["LLM Zero-shot"]["val_acc"]) * 100))
    report.append("")
    
    report.append("3. BERT vs LLM对比")
    report.append("   - 最佳BERT方法: {:.2%}".format(results["BERT（无权重）"]["val_acc"]))
    report.append("   - 最佳LLM方法: {:.2%}".format(results["LLM SFT"]["val_acc"]))
    report.append("   - 结论: LLM SFT微调方法表现最佳，准确率比BERT高{:.1f}个百分点".format(
        (results["LLM SFT"]["val_acc"] - results["BERT（无权重）"]["val_acc"]) * 100))
    report.append("")
    
    report.append("四、方法优缺点分析")
    report.append("-" * 80)
    report.append("")
    report.append("BERT微调方法:")
    report.append("  优点:")
    report.append("    - 训练速度快，收敛稳定")
    report.append("    - 模型参数量小，推理速度快")
    report.append("    - 对标注数据依赖性强，适合有充足标注数据的场景")
    report.append("  缺点:")
    report.append("    - 需要大量标注数据")
    report.append("    - 对新类别需要重新训练")
    report.append("")
    
    report.append("LLM Zero-shot方法:")
    report.append("  优点:")
    report.append("    - 无需训练，即开即用")
    report.append("    - 可以快速适应新类别")
    report.append("    - 适合快速原型验证")
    report.append("  缺点:")
    report.append("    - 准确率较低")
    report.append("    - 输出格式不稳定，存在无法解析的情况")
    report.append("    - 推理速度慢，成本高")
    report.append("")
    
    report.append("LLM SFT微调方法:")
    report.append("  优点:")
    report.append("    - 准确率最高")
    report.append("    - 输出格式稳定，几乎无无法解析情况")
    report.append("    - 结合了LLM的强大语义理解能力")
    report.append("  缺点:")
    report.append("    - 需要标注数据进行微调")
    report.append("    - 推理速度较BERT慢")
    report.append("    - 模型参数量较大")
    report.append("")
    
    report.append("五、应用建议")
    report.append("-" * 80)
    report.append("")
    report.append("1. 追求最高准确率: 推荐LLM SFT微调方法")
    report.append("2. 快速原型验证: 推荐LLM Zero-shot方法")
    report.append("3. 生产环境部署（对速度有要求）: 推荐BERT微调方法")
    report.append("4. 类别不平衡场景: 推荐BERT（有权重）方法")
    report.append("")
    
    report.append("六、数据集类别分布")
    report.append("-" * 80)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels:
        percentage = count / sum(label_counts.values()) * 100
        report.append(f"{label:<10} {count:>6} 条  ({percentage:>5.2f}%)")
    report.append("")
    
    report.append("=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    with open(OUTPUTS_DIR / "comparison_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ 分析报告已保存: {OUTPUTS_DIR / 'comparison_report.txt'}")
    print("\n" + report_text)

def main():
    print("=" * 80)
    print("文本分类方法对比分析")
    print("=" * 80)
    print()
    
    print("加载训练结果...")
    results = load_results()
    
    print("加载标签分布...")
    label_counts, label_map = load_label_distribution()
    
    print("\n生成可视化图表...")
    plot_performance_comparison(results)
    plot_llm_comparison(results)
    plot_training_curves()
    
    print("\n生成分析报告...")
    generate_report(results, label_counts)
    
    print("\n" + "=" * 80)
    print("所有结果已生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
