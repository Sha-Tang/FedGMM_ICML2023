# MNIST测试集运行共识文档

## 1. 明确的需求描述

### 核心目标
在MNIST数据集上运行FedGMM联邦学习实验，验证框架的跨数据集泛化能力。

### 重要发现和修正
**关键发现**: data/mnist/实际上是EMNIST数据集（Extended MNIST），不是标准MNIST
- EMNIST包含手写字母和数字
- 使用Dirichlet分布进行联邦数据分割
- 需要运行generate_data.py生成客户端数据

### 具体需求
- **数据集**: EMNIST (Extended MNIST) 手写字符识别
- **算法**: FedGMM (高斯混合模型联邦学习)
- **规模**: 快速验证，10轮训练
- **环境**: CPU环境，复用FEMNIST成功配置
- **目标**: 验证FedGMM在不同手写识别数据集上的稳定性

## 2. 技术实现方案

### 参数配置
```bash
# 第一步：生成EMNIST联邦数据
cd data/mnist
python generate_data.py \
    --n_tasks 20 \
    --alpha 0.4 \
    --s_frac 0.01 \
    --tr_frac 0.8 \
    --seed 12345

# 第二步：运行FedGMM训练
cd ../..
python run_experiment.py mnist FedGMM \
    --n_learners 3 \
    --n_gmm 3 \
    --n_rounds 10 \
    --bz 64 \
    --lr 0.01 \
    --lr_scheduler multi_step \
    --log_freq 2 \
    --device cpu \
    --optimizer sgd \
    --seed 1234 \
    --logs_dir ./logs \
    --verbose 1
```

### 技术约束
- **环境要求**: 已验证的PyTorch 2.6.0+cpu环境
- **数据管理**: 使用.pkl格式的EMNIST联邦分片
- **模型适配**: 复用28x28→32x32图像转换逻辑
- **日志输出**: logs/mnist/目录下TensorBoard格式

### 集成方案
- **数据流**: EMNIST原始数据 → 联邦分割 → 客户端训练 → 服务端聚合
- **架构复用**: ACGCentralizedAggregator + ACGMixtureClient
- **格式处理**: 复用FEMNIST的28x28灰度图像处理逻辑

## 3. 任务边界限制

### 包含范围
✅ EMNIST数据集下载和联邦分割生成  
✅ FedGMM在EMNIST上的10轮快速训练  
✅ 训练指标收集和日志记录  
✅ 与FEMNIST结果的对比分析  
✅ 完整的实验文档记录  

### 排除范围
❌ EMNIST的详细超参数优化  
❌ 与其他联邦学习算法的详细对比  
❌ 大规模长时间训练实验  
❌ GPU环境的性能对比  
❌ 复杂的可视化分析  

## 4. 验收标准

### 功能验收
1. **数据验收**: EMNIST数据成功生成，客户端数据分片正确
2. **训练验收**: 10轮FedGMM训练无中断完成
3. **指标验收**: 获得训练/测试损失和准确率指标
4. **日志验收**: TensorBoard日志正常生成和记录
5. **对比验收**: 与FEMNIST实验结果进行有意义对比

### 性能验收
- **运行时间**: 整体运行时间 < 20分钟 (包含数据生成)
- **数据生成**: 联邦数据分割 < 5分钟
- **训练执行**: 10轮训练 < 10分钟
- **准确率**: 期望高于FEMNIST结果(EMNIST相对标准化)

### 技术验收  
- **兼容性**: 无需修改核心FedGMM代码
- **稳定性**: 训练过程无异常中断
- **一致性**: 日志格式与FEMNIST实验保持一致
- **可复现**: 相同seed下结果可重现

## 5. 对比价值分析

### FEMNIST vs EMNIST对比维度
- **数据复杂度**: FEMNIST(手写字母+数字) vs EMNIST(扩展字符集)
- **客户端数量**: FEMNIST(35个) vs EMNIST(20个)
- **数据分布**: 不同的自然分割策略
- **收敛速度**: 训练效率对比
- **最终精度**: 分类准确率对比

### 框架验证价值
- **跨数据集稳定性**: 验证FedGMM架构的通用性
- **参数迁移性**: 验证超参数配置的可移植性
- **性能一致性**: 验证算法在不同手写数据上的表现

## 6. 确认所有不确定性已解决

### 已确认的决策
✅ **数据集确认**: 使用EMNIST而非标准MNIST  
✅ **实验规模**: 快速验证策略(10轮训练)  
✅ **客户端数**: 20个客户端，适中规模  
✅ **参数复用**: 基于FEMNIST成功配置  
✅ **对比目标**: 验证跨数据集泛化能力  

### 风险控制
- **数据生成风险**: 使用官方generate_data.py脚本
- **兼容性风险**: 复用已验证的28x28图像处理逻辑
- **训练稳定性风险**: 基于成功的FEMNIST配置
- **时间控制风险**: 采用小规模数据集(s_frac=0.01)

## 7. 项目特性规范对齐

### 代码风格对齐
- 复用现有run_experiment.py入口
- 保持与FEMNIST相同的参数风格
- 使用项目统一的日志格式

### 架构对齐
- 使用相同的ACGCentralizedAggregator架构
- 使用相同的ACGMixtureClient设计
- 保持相同的模型保存和加载策略

### 质量对齐
- 遵循快速验证的时间约束
- 保持高质量的错误处理
- 生成完整的实验文档和对比分析

## 8. 预期输出

### 实验结果
- **训练曲线**: 10轮训练的损失和准确率变化
- **最终指标**: 训练和测试的最终性能数据
- **收敛分析**: 与FEMNIST的收敛速度对比

### 技术产物
- **数据分片**: data/mnist/all_data/客户端数据文件
- **模型保存**: saves/mnist/FedGMM/训练检查点
- **日志文件**: logs/mnist/FedGMM/TensorBoard日志

### 分析报告
- **性能对比**: EMNIST vs FEMNIST详细对比
- **算法验证**: FedGMM跨数据集泛化能力评估
- **技术总结**: 实施经验和改进建议

这个共识确保了MNIST(EMNIST)测试集运行的明确目标和可操作的实施方案。