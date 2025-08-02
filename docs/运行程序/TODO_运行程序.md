# FedGMM运行项目待办事项

## 🔧 技术优化待办

### 高优先级 (建议立即处理)

#### 1. 测试数据集修复
**问题**: FEMNIST测试数据目录为空，影响测试评估准确性
**建议解决方案**:
```bash
# 在data/femnist/目录下重新生成包含测试数据的完整数据集
python generate_data.py --s_frac 0.01 --tr_frac 0.8 --val_frac 0.2 --seed 12345
```
**预期效果**: 获得独立的测试集，提高模型评估的可信度

#### 2. 依赖版本警告处理
**问题**: TorchVision模型加载时出现deprecated参数警告
**建议解决方案**:
```python
# 在run_experiment.py中替换
model = models.resnet18(pretrained=True)
# 改为
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
```
**预期效果**: 消除警告信息，保持代码现代化

### 中优先级 (可选择性处理)

#### 3. 训练精度提升
**问题**: 当前训练准确率较低 (~6%)，主要因为数据量小
**建议解决方案**:
```bash
# 使用更大的数据分片比例
python run_experiment.py femnist FedGMM --s_frac 0.05  # 从0.01提升到0.05
```
**预期效果**: 提高模型性能，获得更有意义的精度指标

#### 4. 超参数调优
**问题**: 使用默认超参数，可能不是最优配置
**建议调优方向**:
- 学习率: 尝试0.001, 0.005, 0.01, 0.05
- 批次大小: 尝试32, 64, 128
- 学习器数量: 尝试1, 3, 5
- GMM组件数: 尝试1, 3, 5

## 🚀 功能扩展待办

### 扩展实验 (可选)

#### 5. 多数据集对比验证
**目标**: 验证FedGMM在不同数据集上的表现
**建议实验**:
```bash
# CIFAR10对比实验
python run_experiment.py cifar10 FedGMM --n_rounds 10

# EMNIST对比实验  
python run_experiment.py emnist FedGMM --n_rounds 10
```

#### 6. 算法性能对比
**目标**: 对比FedGMM与其他联邦学习算法
**建议实验**:
```bash
# FedAvg基线
python run_experiment.py femnist FedAvg --n_rounds 10

# FedProx对比
python run_experiment.py femnist FedProx --n_rounds 10 --mu 0.1
```

## 🔍 监控和可视化

### 建议添加的监控功能

#### 7. TensorBoard可视化增强
**当前状态**: 基本的loss和accuracy记录
**建议增强**:
- 学习率变化曲线
- 客户端参与度统计
- 模型参数分布可视化
- 收敛速度对比图表

#### 8. 实时训练监控
**建议工具**:
```bash
# 启动TensorBoard监控
tensorboard --logdir=./logs --port=6006

# 在浏览器中访问: http://localhost:6006
```

## 📚 文档和维护

### 文档改进待办

#### 9. 完善README文档
**需要添加的内容**:
- Windows环境配置详细步骤
- 常见问题和解决方案
- 不同数据集的使用指南
- 性能调优建议

#### 10. 代码注释完善
**优先级区域**:
- run_experiment.py中的数据格式转换部分
- aggregator.py中的FedGMM特定逻辑
- 新添加的FEMNIST适配代码

## ⚙️ 配置和部署

### 环境配置优化

#### 11. Docker化部署 (可选)
**目标**: 提供一键部署解决方案
**建议创建**:
```dockerfile
# Dockerfile示例
FROM python:3.13-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
```

#### 12. 配置文件标准化
**目标**: 减少命令行参数，提高配置可维护性
**建议创建**: `config/femnist_quick.yaml`
```yaml
experiment: femnist
method: FedGMM
n_learners: 3
n_gmm: 3
n_rounds: 10
batch_size: 64
learning_rate: 0.01
device: cpu
```

## 🧪 测试和验证

### 自动化测试建议

#### 13. 单元测试添加
**建议测试模块**:
- 数据加载和预处理功能
- 模型参数聚合逻辑
- 客户端-服务端通信

#### 14. 集成测试扩展
**建议测试场景**:
- 多轮训练稳定性测试
- 异常恢复能力测试
- 不同硬件环境兼容性测试

## 🎯 性能优化

### 计算效率提升

#### 15. GPU支持验证
**目标**: 在有GPU的环境下验证加速效果
```bash
# GPU版本运行
python run_experiment.py femnist FedGMM --device cuda --n_rounds 10
```

#### 16. 内存优化
**优化方向**:
- 减少重复的模型实例化
- 优化数据批次处理
- 实现梯度检查点机制

## 📋 优先级排序

### 立即处理 (本周内)
1. **测试数据集修复** - 影响实验准确性
2. **依赖版本警告处理** - 代码维护性

### 短期处理 (2-4周内)  
3. **TensorBoard可视化启动** - 提升监控能力
4. **README文档完善** - 提高可用性
5. **训练精度提升实验** - 验证模型效果

### 长期规划 (1-3个月内)
6. **多数据集对比实验** - 扩展验证范围
7. **算法性能对比** - 学术价值提升
8. **Docker化部署** - 降低使用门槛

## 🔧 具体操作指引

### 快速修复命令
```bash
# 1. 重新生成包含测试数据的FEMNIST数据集
cd data/femnist
python generate_data.py --s_frac 0.01 --tr_frac 0.8 --val_frac 0.2 --seed 12345

# 2. 启动TensorBoard监控
cd ../../
tensorboard --logdir=./logs --port=6006

# 3. 运行提升精度的实验
python run_experiment.py femnist FedGMM --n_learners 3 --n_gmm 3 --n_rounds 20 --bz 64 --lr 0.01 --device cpu --verbose 1
```

### 配置文件示例
创建 `configs/quick_validation.json`:
```json
{
  "experiment": "femnist", 
  "method": "FedGMM",
  "n_learners": 3,
  "n_gmm": 3,
  "n_rounds": 10,
  "batch_size": 64,
  "learning_rate": 0.01,
  "device": "cpu",
  "verbose": 1,
  "seed": 1234
}
```

**这些待办事项将帮助您进一步提升FedGMM项目的完整性和实用性。建议根据实际需求和时间安排选择性实施。**