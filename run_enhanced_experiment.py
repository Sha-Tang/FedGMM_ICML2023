#!/usr/bin/env python3
"""
增强版FedGMM+DGC实验脚本
包含详细的模型上传下载大小打印功能

使用方法:
python run_enhanced_experiment.py cifar10 FedGMM --use_dgc --compress_ratio 0.1 --warmup_rounds 3
"""

import os
import sys

# 确保当前目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入原始的run_experiment模块
import run_experiment

def main():
    """
    运行增强版实验，带有详细的通信统计打印
    """
    print("🚀 启动增强版FedGMM+DGC实验")
    print("📊 将显示详细的上传下载模型大小信息")
    print("-" * 60)
    
    # 直接调用原始实验脚本
    run_experiment.main()

if __name__ == "__main__":
    main() 