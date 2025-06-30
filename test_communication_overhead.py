#!/usr/bin/env python3
# test_communication_overhead.py - 测试通信开销指标的演示脚本

"""
测试FedGMM通信开销指标功能
演示如何追踪和可视化累计通信开销

运行方式：
python test_communication_overhead.py
"""

import torch
import numpy as np
from utils.args import parse_args


def demo_communication_overhead_calculation():
    """演示通信开销计算逻辑"""
    print("📊 通信开销计算演示")
    print("=" * 50)
    
    # 模拟参数数据
    print("🔧 创建模拟参数数据...")
    
    # 模拟原始模型参数 (假设10万个参数)
    original_params = torch.randn(100000)
    original_size = original_params.numel()
    print(f"   原始参数量: {original_size:,}")
    
    # 模拟不同压缩比的效果
    compression_scenarios = [
        {'name': '无压缩', 'ratio': 1.0},
        {'name': '轻度压缩', 'ratio': 0.7},
        {'name': '中等压缩', 'ratio': 0.1},
        {'name': '高度压缩', 'ratio': 0.01},
    ]
    
    print("\n📈 不同压缩比的通信开销对比:")
    print(f"{'压缩类型':<12} {'压缩比':<8} {'上传参数量':<12} {'节省百分比':<10}")
    print("-" * 50)
    
    for scenario in compression_scenarios:
        uploaded_size = int(original_size * scenario['ratio'])
        savings_pct = (1 - scenario['ratio']) * 100
        
        print(f"{scenario['name']:<12} {scenario['ratio']:<8.0%} {uploaded_size:<12,} {savings_pct:<10.1f}%")
    
    print("\n💡 示例说明:")
    print("   - 原始参数100,000个，压缩比70%时，上传70,000个模型参数")
    print("   - 压缩比1%时，只需上传1,000个模型参数，节省99%参数传输")
    print("   - 注意：现在只统计模型参数量，不包括压缩索引和元数据开销")


def demo_cumulative_tracking():
    """演示累计追踪逻辑"""
    print("\n📈 累计通信开销追踪演示")
    print("=" * 50)
    
    # 模拟多轮训练的通信开销
    rounds = 10
    original_size_per_round = 100000
    compression_ratio = 0.01  # 1%压缩
    
    total_original = 0
    total_uploaded = 0
    
    print(f"{'轮次':<4} {'原始大小':<10} {'上传大小':<10} {'累计原始':<12} {'累计上传':<12} {'累计节省':<10}")
    print("-" * 70)
    
    for round_num in range(1, rounds + 1):
        # 每轮的通信量
        round_original = original_size_per_round
        
        # 前3轮预热，不压缩
        if round_num <= 3:
            round_uploaded = round_original
        else:
            round_uploaded = int(round_original * compression_ratio)
        
        # 累计统计
        total_original += round_original
        total_uploaded += round_uploaded
        
        # 计算累计节省
        cumulative_savings = (1 - total_uploaded / total_original) * 100
        
        print(f"{round_num:<4} {round_original:<10,} {round_uploaded:<10,} {total_original:<12,} {total_uploaded:<12,} {cumulative_savings:<10.1f}%")
    
    print(f"\n📊 最终统计:")
    print(f"   总原始参数量: {total_original:,}")
    print(f"   总上传参数量: {total_uploaded:,}")
    print(f"   总体节省比例: {(1 - total_uploaded / total_original) * 100:.1f}%")


def demo_tensorboard_metrics():
    """演示TensorBoard指标结构"""
    print("\n📊 TensorBoard指标展示")
    print("=" * 50)
    
    print("🎯 新增的通信开销指标:")
    
    metrics = [
        {
            'category': 'Communication',
            'metrics': [
                'classifier_original_size - 分类器原始参数大小',
                'classifier_uploaded_size - 分类器实际上传大小', 
                'classifier_size_ratio - 分类器上传比例',
                'autoencoder_original_size - 自编码器原始参数大小',
                'autoencoder_uploaded_size - 自编码器实际上传大小',
                'autoencoder_size_ratio - 自编码器上传比例',
                'total_original_params - 累计原始参数总量',
                'total_uploaded_params - 累计上传参数总量',
                'cumulative_overhead - 累计通信开销',
                'total_savings - 累计节省参数量',
                'savings_ratio - 累计节省比例',
                'overall_compression_ratio - 总体压缩比',
                'summary_total_savings_ratio - 汇总节省比例',
                'summary_overall_compression - 汇总压缩比',
                'summary_total_rounds - 汇总轮次数'
            ]
        },
        {
            'category': 'Compression',
            'metrics': [
                'ratio - 压缩比',
                'classifier_ratio - 分类器压缩比',
                'autoencoder_ratio - 自编码器压缩比',
                'savings_pct - 节省百分比',
                'classifier_savings_pct - 分类器节省百分比',
                'autoencoder_savings_pct - 自编码器节省百分比'
            ]
        }
    ]
    
    for category in metrics:
        print(f"\n📁 {category['category']} 类别:")
        for metric in category['metrics']:
            print(f"   📊 {metric}")
    
    print(f"\n🔍 使用方法:")
    print(f"   1. 启动TensorBoard: tensorboard --logdir=logs/")
    print(f"   2. 在浏览器中打开: http://localhost:6006")
    print(f"   3. 查看 Communication 和 Compression 标签页")
    print(f"   4. 重点关注: cumulative_overhead, savings_ratio, overall_compression_ratio")


def demo_practical_example():
    """演示实际使用示例"""
    print("\n🚀 实际使用示例")
    print("=" * 50)
    
    example_commands = [
        {
            'name': '基础监控 (1%压缩)',
            'cmd': '''python run_experiment.py cifar10 FedGMM \\
    --n_learners 3 --n_gmm 3 --n_rounds 20 \\
    --use_dgc --topk_ratio 0.01 \\
    --logs_dir ./logs/cifar10/communication_demo''',
            'expected': '预期节省99%通信量'
        },
        {
            'name': '高压缩监控 (0.1%压缩)',
            'cmd': '''python run_experiment.py cifar10 FedGMM \\
    --n_learners 3 --n_gmm 3 --n_rounds 20 \\
    --use_dgc --topk_ratio 0.001 \\
    --logs_dir ./logs/cifar10/high_compression''',
            'expected': '预期节省99.9%通信量'
        }
    ]
    
    for i, example in enumerate(example_commands, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   命令: {example['cmd']}")
        print(f"   结果: {example['expected']}")
    
    print(f"\n📊 运行后你将看到:")
    print(f"   控制台输出:")
    print(f"   📊 Round 5 [classifier] Model Parameter Summary:")
    print(f"      Original params: 100,000 → Compressed params: 1,000 (1.0%)")
    print(f"      Current round savings: 99,000 params (99.0%)")
    print(f"      Total param savings: 495,000 params (99.0%)")
    print(f"      Note: All savings metrics are guaranteed to be positive")


def main():
    """主函数"""
    print("🔄 FedGMM 通信开销指标演示")
    print("=" * 60)
    
    # 基础计算演示
    demo_communication_overhead_calculation()
    
    # 累计追踪演示
    demo_cumulative_tracking()
    
    # TensorBoard指标演示
    demo_tensorboard_metrics()
    
    # 实际使用示例
    demo_practical_example()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print("💡 现在你可以运行实际实验并在TensorBoard中查看通信开销指标了！")


if __name__ == "__main__":
    main() 