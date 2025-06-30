# 🔄 FedGMM 通信压缩功能使用指南

## 📋 概述

本文档介绍如何在FedGMM项目中启用和使用通信压缩功能，以减少联邦学习过程中的通信开销。

## 🛠️ 核心组件

### 1. 压缩工具模块 (`utils/compression.py`)

提供了完整的Top-K压缩算法实现，支持DGC残差补偿：

- **CommunicationCompressor**: 核心压缩器类
- **create_compressor()**: 根据参数创建压缩器实例
- **should_compress()**: 判断是否压缩当前轮次
- **should_reset_residual()**: 判断是否重置残差缓存

### 2. ACGLearnersEnsemble 压缩支持

为`ACGLearnersEnsemble`类添加了以下压缩方法：

- `enable_compression(args)`: 启用压缩功能
- `get_flat_model_params()` / `set_flat_model_params()`: 参数展平和重建
- `get_compressed_params()` / `set_compressed_params()`: 压缩和解压缩
- `fit_epochs_with_compression()`: 带压缩功能的训练方法

## 🚀 使用方法

### 步骤1: 启用压缩参数

在命令行中添加压缩相关参数：

```bash
python run_experiment.py \
    --experiment_name "fedgmm_compressed" \
    --use_dgc \
    --topk_ratio 0.01 \
    --topk_strategy "magnitude" \
    --warmup_rounds 5 \
    --force_upload_every 10 \
    --n_rounds 50
```

### 步骤2: 在代码中启用压缩

```python
# 在客户端初始化时启用压缩
learners_ensemble = ACGLearnersEnsemble(learners, embedding_dim, autoencoder, n_gmm)
learners_ensemble.enable_compression(args)

# 使用带压缩功能的训练方法
compressed_updates = learners_ensemble.fit_epochs_with_compression(
    iterator=train_loader,
    n_epochs=args.local_learning_rate,
    weights=sample_weights,
    current_round=communication_round
)
```

### 步骤3: 处理压缩结果

```python
# 检查是否为压缩数据
if isinstance(compressed_updates, dict) and compressed_updates.get('compressed', False):
    # 处理压缩数据
    print(f"压缩比: {compressed_updates['compression_ratio']:.1%}")
    
    # 在服务端解压缩
    decompressed = learners_ensemble.set_compressed_params(compressed_updates)
else:
    # 处理完整数据
    decompressed = compressed_updates
```

## ⚙️ 参数配置详解

### 压缩控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_dgc` | bool | False | 是否启用通信压缩 |
| `--topk_ratio` | float | 0.01 | Top-K压缩比例(0-1) |
| `--topk_strategy` | str | 'magnitude' | 压缩策略('magnitude'或'relative') |
| `--warmup_rounds` | int | 5 | 预热轮数，期间不压缩 |
| `--force_upload_every` | int | 10 | 强制全量上传间隔轮数 |

### 压缩策略说明

- **magnitude**: 按参数绝对值大小选择Top-K
- **relative**: 按相对变化选择Top-K（当前简化为magnitude）

## 📊 压缩效果监控

### 获取压缩统计信息

```python
# 获取压缩统计
stats = learners_ensemble.get_compression_stats()
print(f"压缩轮数: {stats['compressed_rounds']}")
print(f"完整上传轮数: {stats['full_upload_rounds']}")
print(f"平均压缩比: {stats['avg_compression_ratio']:.1%}")
```

### 统计信息字段

```python
{
    'compression_enabled': True,
    'total_rounds': 50,
    'compressed_rounds': 40,
    'full_upload_rounds': 10,
    'avg_compression_ratio': 0.01,
    'original_size': 1000000,
    'compressed_size': 10000,
    'topk_ratio': 0.01,
    'strategy': 'magnitude'
}
```

## 🔧 高级用法

### 手动控制压缩流程

```python
# 1. 获取压缩配置
compression_info = learners_ensemble.get_compressed_params(current_round=15)

# 2. 执行标准训练
client_updates = learners_ensemble.fit_epochs(iterator, n_epochs, weights)

# 3. 手动应用压缩
client_updates_tensor = torch.tensor(client_updates)
compressed_result = learners_ensemble.apply_compression_to_updates(
    client_updates_tensor, 
    compression_info
)

# 4. 递增轮次
learners_ensemble.increment_round()
```

### 残差补偿机制

DGC算法会自动处理残差补偿：

- **残差累积**: 未压缩的参数累积为残差
- **残差补偿**: 下次压缩时添加历史残差
- **定期重置**: 强制上传轮会重置残差缓存

## 🎯 最佳实践

### 1. 参数调优建议

- **topk_ratio**: 建议从0.01开始，根据收敛效果调整
- **warmup_rounds**: 模型稳定前避免压缩，建议5-10轮
- **force_upload_every**: 平衡压缩效果和收敛稳定性，建议10-20轮

### 2. 使用场景

✅ **适用场景**:
- 网络带宽受限的分布式环境
- 大模型联邦学习
- 需要降低通信成本的场景

❌ **不适用场景**:
- 本地训练或高速网络环境
- 模型参数量很小的情况
- 对收敛精度要求极高的场景

### 3. 性能监控

```python
# 定期输出压缩统计
if communication_round % 10 == 0:
    stats = learners_ensemble.get_compression_stats()
    logger.info(f"Round {communication_round} - Compression Stats: {stats}")
```

## 🔍 故障排除

### 常见问题

1. **ImportError**: 确保`utils/compression.py`在Python路径中
2. **参数错误**: 检查压缩参数是否正确传递给`args`对象
3. **内存溢出**: 大模型压缩时注意内存使用，考虑分批处理

### 调试技巧

```python
# 启用详细压缩日志
learners_ensemble.enable_compression(args)
# 会输出压缩配置和每轮压缩统计

# 检查压缩器状态
if learners_ensemble.compressor:
    print(f"压缩器状态: {learners_ensemble.compressor.get_stats()}")
```

## 📈 实验建议

### 对比实验设置

```bash
# 基准实验（无压缩）
python run_experiment.py --experiment_name "baseline"

# 压缩实验（1%压缩率）
python run_experiment.py --experiment_name "compressed_1pct" \
    --use_dgc --topk_ratio 0.01 --warmup_rounds 5 --force_upload_every 10

# 压缩实验（0.1%压缩率）
python run_experiment.py --experiment_name "compressed_01pct" \
    --use_dgc --topk_ratio 0.001 --warmup_rounds 5 --force_upload_every 10
```

### 性能评估指标

- **通信量减少**: `(1 - avg_compression_ratio) * 100%`
- **收敛速度**: 达到目标精度所需轮数
- **最终精度**: 训练结束时的模型性能

通过以上配置和使用方法，您可以在FedGMM项目中有效地启用和使用通信压缩功能，显著降低联邦学习的通信开销。 