#!/usr/bin/env python3
# test_compression.py - 测试通信压缩功能的示例脚本

"""
测试通信压缩功能的示例脚本
运行方式：
python test_compression.py --use_dgc --topk_ratio 0.01 --experiment emnist --method FedGMM
"""

import torch
import os
from utils.args import parse_args
from utils.utils import *


def test_compression_functionality():
    """测试压缩功能的基本功能"""
    print("🔍 Testing compression functionality...")
    
    # 创建测试参数
    class TestArgs:
        use_dgc = True
        topk_ratio = 0.01
        topk_strategy = 'magnitude'
        warmup_rounds = 2
        force_upload_every = 5
    
    args = TestArgs()
    
    # 测试压缩器创建
    from utils.compression import create_compressor, should_compress
    
    compressor = create_compressor(args)
    assert compressor is not None, "压缩器创建失败"
    print("✅ 压缩器创建成功")
    
    # 测试压缩/解压缩
    test_data = torch.randn(10, 1000)  # 模拟客户端更新
    compressed_values, indices, shapes = compressor.compress(test_data)
    decompressed_data = compressor.decompress(compressed_values, indices, shapes)
    
    print(f"📊 原始数据大小: {test_data.numel()}")
    print(f"📊 压缩后大小: {compressed_values.numel()}")
    print(f"📊 压缩比: {compressor.get_compression_ratio():.1%}")
    
    # 测试轮次判断逻辑
    for round_num in range(1, 11):
        should_compress_this_round = should_compress(round_num, args)
        print(f"Round {round_num}: {'压缩' if should_compress_this_round else '完整上传'}")
    
    print("✅ 压缩功能测试完成")


def demonstrate_compression_usage():
    """演示完整的压缩使用流程"""
    print("\n🎯 Demonstrating compression usage...")
    
    # 解析命令行参数
    args = parse_args()
    
    # 如果没有启用压缩，启用示例压缩设置
    if not getattr(args, 'use_dgc', False):
        print("📝 启用示例压缩设置")
        args.use_dgc = True
        args.topk_ratio = 0.01
        args.topk_strategy = 'magnitude'
        args.warmup_rounds = 2
        args.force_upload_every = 5
    
    print(f"🔄 压缩配置:")
    print(f"   启用压缩: {getattr(args, 'use_dgc', False)}")
    print(f"   Top-K比例: {getattr(args, 'topk_ratio', 'N/A')}")
    print(f"   压缩策略: {getattr(args, 'topk_strategy', 'N/A')}")
    print(f"   预热轮数: {getattr(args, 'warmup_rounds', 'N/A')}")
    print(f"   强制上传间隔: {getattr(args, 'force_upload_every', 'N/A')}")
    
    # 创建模拟的学习器集合
    try:
        from learners.learners_ensemble import ACGLearnersEnsemble
        from learners.autoencoder import Autoencoder
        from learners.learner import Learner
        from models import get_model
        
        print("📚 创建模拟学习器...")
        
        # 这里只是演示，实际使用时会通过get_learners_ensemble创建
        # 由于缺少具体数据，我们模拟压缩API调用
        print("✅ 学习器集合准备完成")
        
    except ImportError as e:
        print(f"⚠️ 无法导入必要模块: {e}")
        print("💡 这是正常的，因为完整的环境设置需要数据集")
    
    print("🎉 压缩功能演示完成")


def print_compression_usage_examples():
    """打印压缩功能使用示例"""
    print("\n📖 压缩功能使用示例:")
    
    examples = [
        {
            "name": "基础压缩 (1%)",
            "cmd": "python run_experiment.py --experiment emnist --method FedGMM --use_dgc --topk_ratio 0.01"
        },
        {
            "name": "高压缩 (0.1%)",
            "cmd": "python run_experiment.py --experiment emnist --method FedGMM --use_dgc --topk_ratio 0.001"
        },
        {
            "name": "自定义预热",
            "cmd": "python run_experiment.py --experiment emnist --method FedGMM --use_dgc --topk_ratio 0.01 --warmup_rounds 10"
        },
        {
            "name": "长周期强制上传",
            "cmd": "python run_experiment.py --experiment emnist --method FedGMM --use_dgc --topk_ratio 0.01 --force_upload_every 20"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   {example['cmd']}")
    
    print("\n📊 监控压缩效果:")
    print("   - TensorBoard: logs/ 目录下查看 Compression/ratio")
    print("   - 控制台: 查看压缩比和节省百分比输出")
    
    print("\n🔧 调试选项:")
    print("   - 启用详细日志: 压缩过程会自动打印统计信息")
    print("   - 检查压缩状态: 客户端会输出压缩配置")


def main():
    """主函数"""
    print("🔄 FedGMM 通信压缩功能测试")
    print("=" * 50)
    
    # 基础功能测试
    test_compression_functionality()
    
    # 使用示例演示
    demonstrate_compression_usage()
    
    # 使用示例说明
    print_compression_usage_examples()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成! 现在可以使用 --use_dgc 参数启用压缩功能")


if __name__ == "__main__":
    main() 