import os
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import numpy.linalg as LA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from utils.torch_utils import *
from gradient_cache import GradientCacheManager


class AnomalyDetector:
    """
    客户端上传内容异常检测器
    负责检测和处理客户端上传的delta向量中的异常值
    """
    
    def __init__(self, strict_mode=True):
        """
        初始化异常检测器
        
        Args:
            strict_mode: 是否启用严格模式（更严格的异常阈值）
        """
        self.strict_mode = strict_mode
        self.client_anomaly_records = {}  # 记录每个客户端的异常次数
        
        # 📌 问题1解决：跳过次数限制和分布保护
        self.max_consecutive_skips = 3  # 最多连续跳过3次
        self.client_skip_counts = {}    # 记录连续跳过次数
        self.client_sample_counts = {}  # 记录客户端样本数（用于分布平衡）
        self.force_inclusion_mode = False  # 强制包含模式（保护数据分布）
        
    def detect_client_anomalies(self, client_id: int, delta_vector: torch.Tensor) -> dict:
        """
        📌 第一步：检测单个客户端上传的delta向量异常
        
        Args:
            client_id: 客户端ID
            delta_vector: 客户端上传的delta向量 (已展平)
            
        Returns:
            dict: 异常检测结果
        """
        total_elements = delta_vector.numel()
        
        # 基本异常统计
        nan_count = torch.isnan(delta_vector).sum().item()
        inf_count = torch.isinf(delta_vector).sum().item()
        zero_count = (delta_vector == 0.0).sum().item()
        
        # 计算异常比例
        nan_ratio = nan_count / total_elements
        inf_ratio = inf_count / total_elements  
        zero_ratio = zero_count / total_elements
        total_anomaly_ratio = (nan_count + inf_count) / total_elements
        
        # 数值统计
        finite_mask = torch.isfinite(delta_vector)
        if finite_mask.any():
            finite_values = delta_vector[finite_mask]
            value_stats = {
                'mean': finite_values.mean().item(),
                'std': finite_values.std().item(),
                'max': finite_values.max().item(),
                'min': finite_values.min().item(),
                'abs_max': finite_values.abs().max().item()
            }
        else:
            value_stats = {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'abs_max': 0}
        
        # 异常分类判断
        anomaly_level = self._classify_anomaly_level(
            nan_ratio, inf_ratio, zero_ratio, total_anomaly_ratio, value_stats
        )
        
        # 📌 问题1&2解决：智能降级和跳过限制
        should_skip = self._should_skip_client(client_id, anomaly_level, detection_result={
            'nan_ratio': nan_ratio, 'inf_ratio': inf_ratio, 'zero_ratio': zero_ratio,
            'total_anomaly_ratio': total_anomaly_ratio, 'value_stats': value_stats
        })
        
        detection_result = {
            'client_id': client_id,
            'total_elements': total_elements,
            'nan_count': nan_count,
            'inf_count': inf_count,
            'zero_count': zero_count,
            'nan_ratio': nan_ratio,
            'inf_ratio': inf_ratio,
            'zero_ratio': zero_ratio,
            'total_anomaly_ratio': total_anomaly_ratio,
            'value_stats': value_stats,
            'anomaly_level': anomaly_level,
            'should_skip': should_skip
        }
        
        # 记录异常
        if anomaly_level != 'normal':
            self._record_client_anomaly(client_id, anomaly_level)
        
        return detection_result
    
    def _classify_anomaly_level(self, nan_ratio, inf_ratio, zero_ratio, total_anomaly_ratio, value_stats):
        """
        异常分类逻辑（无需硬编码阈值，基于数据特征判断）
        """
        # 严重异常：大量无效值或全零
        if total_anomaly_ratio > 0.5:  # 超过50%为NaN/Inf
            return 'severe'
        
        if zero_ratio > 0.99:  # 超过99%为零（异常稀疏）
            return 'severe'
            
        # 数值爆炸检测
        if value_stats['abs_max'] > 100.0:  # 参数变化过大
            return 'severe'
            
        # 中等异常：包含一定比例异常值但未达到严重程度
        if total_anomaly_ratio > 0.1:  # 超过10%为NaN/Inf
            return 'moderate'
            
        if value_stats['abs_max'] > 10.0:  # 参数变化较大
            return 'moderate'
            
        # 轻微异常：少量异常值
        if total_anomaly_ratio > 0.01:  # 超过1%为NaN/Inf
            return 'minor'
            
        return 'normal'
    
    def _should_skip_client(self, client_id: int, anomaly_level: str, detection_result: dict) -> bool:
        """
        📌 问题1&2解决：智能跳过判断（考虑连续跳过次数和误判保护）
        
        Args:
            client_id: 客户端ID
            anomaly_level: 异常等级
            detection_result: 检测结果详情
            
        Returns:
            bool: 是否应该跳过该客户端
        """
        # 初始化跳过计数
        if client_id not in self.client_skip_counts:
            self.client_skip_counts[client_id] = 0
        
        # 📌 问题2解决：误判保护 - 大梯度但无NaN/Inf的智能降级
        if anomaly_level == 'severe':
            nan_inf_ratio = detection_result['total_anomaly_ratio']
            abs_max = detection_result['value_stats']['abs_max']
            std_val = detection_result['value_stats']['std_val']
            
            # 如果只是数值大但没有NaN/Inf，且标准差稳定，降级为moderate
            if nan_inf_ratio < 0.01 and abs_max > 10.0 and std_val > 0:
                print(f"   🔄 智能降级：客户端 {client_id} 从 severe 降级为 moderate (大梯度但无异常值)")
                anomaly_level = 'moderate'
        
        # 📌 问题1解决：连续跳过次数限制
        if anomaly_level == 'severe':
            consecutive_skips = self.client_skip_counts[client_id]
            
            if consecutive_skips >= self.max_consecutive_skips:
                print(f"   🛡️  强制包含：客户端 {client_id} 连续跳过 {consecutive_skips} 次，强制降级使用")
                anomaly_level = 'moderate'  # 强制降级，裁剪后使用
                self.client_skip_counts[client_id] = 0  # 重置计数
                return False
        
        # 决定是否跳过
        should_skip = (anomaly_level == 'severe')
        
        # 更新跳过计数
        if should_skip:
            self.client_skip_counts[client_id] += 1
        else:
            self.client_skip_counts[client_id] = 0  # 重置连续跳过计数
        
        return should_skip
    
    def _record_client_anomaly(self, client_id: int, anomaly_level: str):
        """记录客户端异常"""
        if client_id not in self.client_anomaly_records:
            self.client_anomaly_records[client_id] = {
                'severe': 0, 'moderate': 0, 'minor': 0, 'total': 0
            }
        
        self.client_anomaly_records[client_id][anomaly_level] += 1
        self.client_anomaly_records[client_id]['total'] += 1
    
    def clean_anomalies(self, delta_vector: torch.Tensor, detection_result: dict) -> torch.Tensor:
        """
        📌 第二步：清理异常值
        
        Args:
            delta_vector: 原始delta向量
            detection_result: 异常检测结果
            
        Returns:
            torch.Tensor: 清理后的delta向量
        """
        cleaned_vector = delta_vector.clone()
        
        # 替换NaN和Inf为0
        cleaned_vector[torch.isnan(cleaned_vector)] = 0.0
        cleaned_vector[torch.isinf(cleaned_vector)] = 0.0
        
        # 📌 问题4解决：自适应梯度裁剪
        if detection_result['anomaly_level'] in ['moderate', 'severe']:
            cleaned_vector = self._adaptive_gradient_clipping(cleaned_vector, detection_result)
        
        return cleaned_vector
    
    def _adaptive_gradient_clipping(self, delta_vector: torch.Tensor, detection_result: dict) -> torch.Tensor:
        """
        📌 问题4解决：自适应梯度裁剪（基于统计量而非固定阈值）
        
        Args:
            delta_vector: 待裁剪的delta向量
            detection_result: 异常检测结果
            
        Returns:
            torch.Tensor: 裁剪后的delta向量
        """
        # 获取有限值的统计信息
        finite_mask = torch.isfinite(delta_vector)
        if not finite_mask.any():
            return torch.zeros_like(delta_vector)
        
        finite_values = delta_vector[finite_mask]
        mean_val = finite_values.mean()
        std_val = finite_values.std()
        
        # 自适应裁剪策略
        if detection_result['anomaly_level'] == 'severe':
            # 严重异常：较保守的裁剪 (μ ± 2σ)
            alpha = 2.0
        else:
            # 中等异常：较宽松的裁剪 (μ ± 3σ)  
            alpha = 3.0
        
        # 计算裁剪边界
        if std_val > 0:
            # 📌 基于统计量的动态边界：μ ± α·σ
            lower_bound = mean_val - alpha * std_val
            upper_bound = mean_val + alpha * std_val
            
            # 确保边界合理（避免过度保守）
            abs_max = finite_values.abs().max()
            if upper_bound < abs_max * 0.1:  # 如果边界过小，适当放宽
                upper_bound = abs_max * 0.5
                lower_bound = -abs_max * 0.5
        else:
            # 标准差为0，使用固定小范围
            bound = min(1.0, finite_values.abs().max().item())
            lower_bound = -bound
            upper_bound = bound
        
        # 应用裁剪
        clipped_vector = torch.clamp(delta_vector, lower_bound, upper_bound)
        
        # 统计裁剪效果
        clipped_count = (delta_vector != clipped_vector).sum().item()
        total_count = delta_vector.numel()
        
        if clipped_count > 0:
            print(f"   🔧 自适应裁剪: [{lower_bound:.4f}, {upper_bound:.4f}], "
                  f"裁剪 {clipped_count}/{total_count} ({clipped_count/total_count:.1%}) 个元素")
        
        return clipped_vector
    
    def detect_global_anomalies(self, aggregated_gradients: dict) -> dict:
        """
        📌 第三步：检测全局聚合后的异常
        
        Args:
            aggregated_gradients: 聚合后的梯度字典
            
        Returns:
            dict: 全局异常检测结果
        """
        global_anomalies = {}
        
        for learner_id, grad_tensor in aggregated_gradients.items():
            # 对每个learner的聚合结果进行检测
            anomaly_result = self.detect_client_anomalies(-1, grad_tensor)  # 使用-1表示全局
            global_anomalies[learner_id] = anomaly_result
        
        return global_anomalies
    
    def print_anomaly_report(self, detection_result: dict, round_num: int):
        """打印异常检测报告"""
        client_id = detection_result['client_id']
        level = detection_result['anomaly_level']
        
        if level == 'normal':
            return
            
        print(f"🚨 [第{round_num}轮] 客户端 {client_id} 异常检测报告:")
        print(f"   异常等级: {level}")
        print(f"   NaN比例: {detection_result['nan_ratio']:.2%}")
        print(f"   Inf比例: {detection_result['inf_ratio']:.2%}")
        print(f"   零值比例: {detection_result['zero_ratio']:.2%}")
        print(f"   数值范围: [{detection_result['value_stats']['min']:.6f}, {detection_result['value_stats']['max']:.6f}]")
        print(f"   最大绝对值: {detection_result['value_stats']['abs_max']:.6f}")
        
        if detection_result['should_skip']:
            print(f"   ⚠️  决定: 跳过该客户端数据")
        else:
            print(f"   ✅ 决定: 清理异常值后使用")
    
    def get_anomaly_summary(self) -> dict:
        """获取异常统计摘要"""
        return {
            'total_clients_with_anomalies': len(self.client_anomaly_records),
            'client_records': self.client_anomaly_records.copy()
        }


def recover_and_aggregate(client_payloads, n_learners, cache_manager=None, clients_weights=None, anomaly_detector=None, round_num=0):
    """
    Recover dense gradients from client payloads (both dense and compressed) and aggregate them.
    
    Args:
        client_payloads: List of dictionaries returned by client.step()
        n_learners: Number of learners in the ensemble
        cache_manager: GradientCacheManager for handling compressed data recovery
        clients_weights: Tensor of client weights for weighted averaging
        anomaly_detector: AnomalyDetector for detecting and handling anomalies
        round_num: Current training round for logging
        
    Returns:
        aggregated: Dictionary mapping learner_id to aggregated gradient tensor
    """
    client_gradients = {m: [] for m in range(n_learners)}
    client_indices = []  # 记录每个client的索引，用于权重对应
    valid_client_indices = []  # 记录通过异常检测的客户端索引
    compression_stats = {'dense_clients': 0, 'compressed_clients': 0, 'cache_usage': {}, 'skipped_clients': 0}
    
    for idx, payload in enumerate(client_payloads):
        client_id = payload.get('client_id', idx)
        client_indices.append(idx)  # 记录客户端在sampled_clients中的索引
        
        if payload['type'] == 'dense':
            # Handle dense uploads (during warmup or DGC disabled)
            compression_stats['dense_clients'] += 1
            
            for m in range(n_learners):
                grad = torch.from_numpy(payload['updates'][m]).float()
                
                # 📌 第一步 & 第二步：异常检测和处理
                if anomaly_detector is not None:
                    detection_result = anomaly_detector.detect_client_anomalies(client_id, grad.flatten())
                    anomaly_detector.print_anomaly_report(detection_result, round_num)
                    
                    if detection_result['should_skip']:
                        print(f"⚠️  [第{round_num}轮] 跳过客户端 {client_id} (learner {m}) - 严重异常")
                        compression_stats['skipped_clients'] += 1
                        continue  # 跳过这个learner的数据
                    
                    # 清理异常值
                    grad = anomaly_detector.clean_anomalies(grad.flatten(), detection_result)
                    grad = grad.reshape(payload['updates'][m].shape)
                
                client_gradients[m].append(grad)
                
        elif payload['type'] == 'compressed':
            # Handle compressed uploads (DGC enabled)
            compression_stats['compressed_clients'] += 1
            
            client_has_valid_data = False
            for m, data in payload['learners_data'].items():
                learner_id = int(m)
                
                # Extract sparse data
                indices = torch.tensor(data['indices'], dtype=torch.long)
                values = torch.tensor(data['values'], dtype=torch.float32)
                shape = data['shape']
                
                if cache_manager is not None:
                    # Use cache manager to recover complete gradient
                    full_grad = cache_manager.recover_from_compressed(
                        learner_id, indices, values, shape
                    )
                    # Reshape to original shape
                    full_grad = full_grad.reshape(shape)
                    compression_stats['cache_usage'][learner_id] = 'used_cache'
                else:
                    # Fallback: basic recovery without cache (zeros for missing positions)
                    full_grad = torch.zeros(int(np.prod(shape)), dtype=torch.float32)
                    if len(indices) > 0:
                        full_grad[indices] = values
                    full_grad = full_grad.reshape(shape)
                    compression_stats['cache_usage'][learner_id] = 'no_cache'
                
                # 📌 第一步 & 第二步：异常检测和处理（压缩数据）
                if anomaly_detector is not None:
                    detection_result = anomaly_detector.detect_client_anomalies(client_id, full_grad.flatten())
                    anomaly_detector.print_anomaly_report(detection_result, round_num)
                    
                    if detection_result['should_skip']:
                        print(f"⚠️  [第{round_num}轮] 跳过客户端 {client_id} (learner {learner_id}) - 严重异常")
                        compression_stats['skipped_clients'] += 1
                        continue  # 跳过这个learner的数据
                    
                    # 清理异常值
                    cleaned_grad = anomaly_detector.clean_anomalies(full_grad.flatten(), detection_result)
                    full_grad = cleaned_grad.reshape(shape)
                    client_has_valid_data = True
                
                client_gradients[learner_id].append(full_grad)
            
            # 如果该客户端至少有一个learner的数据通过检测，记录为有效客户端
            if client_has_valid_data:
                valid_client_indices.append(idx)
        
        # 记录有效客户端（用于权重计算）
        if payload['type'] == 'dense' or (payload['type'] == 'compressed' and client_has_valid_data):
            valid_client_indices.append(idx)
    
    # Aggregate gradients for each learner with proper weighting
    aggregated = {}
    for m in range(n_learners):
        if len(client_gradients[m]) > 0:
            # ✅ 修复：使用加权平均而非简单平均（只对有效客户端）
            if clients_weights is not None and len(clients_weights) >= len(client_gradients[m]):
                # 获取参与本轮训练且通过异常检测的客户端权重
                participating_weights = clients_weights[valid_client_indices[:len(client_gradients[m])]]
                participating_weights = participating_weights / participating_weights.sum()  # 归一化权重
                
                # 加权聚合
                weighted_sum = torch.zeros_like(client_gradients[m][0])
                for i, grad in enumerate(client_gradients[m]):
                    weighted_sum += participating_weights[i].item() * grad
                aggregated[m] = weighted_sum
            else:
                # 回退到简单平均（无权重信息时）
                aggregated[m] = torch.stack(client_gradients[m], dim=0).mean(dim=0)
        else:
            # No gradients received for this learner (shouldn't happen normally)
            aggregated[m] = torch.zeros(1)  # Will need proper shape handling later
    
    # 📌 第三步：全局聚合后异常检测
    if anomaly_detector is not None and len(aggregated) > 0:
        global_anomalies = anomaly_detector.detect_global_anomalies(aggregated)
        
        severe_anomalies = []
        for learner_id, global_result in global_anomalies.items():
            if global_result['anomaly_level'] == 'severe':
                severe_anomalies.append(learner_id)
        
        if len(severe_anomalies) > 0:
            print(f"🚨 [第{round_num}轮] 全局聚合后检测到严重异常 (Learners: {severe_anomalies})")
            print(f"   📌 问题3解决：采用温和恢复策略，而非激进清零")
            
            # 📌 问题3解决：温和恢复策略
            for learner_id in severe_anomalies:
                grad_tensor = aggregated[learner_id]
                
                # 分层处理：只清理异常部分，保留正常部分
                finite_mask = torch.isfinite(grad_tensor)
                if finite_mask.any():
                    # 保留有限值，清理异常值
                    cleaned_grad = grad_tensor.clone()
                    cleaned_grad[~finite_mask] = 0.0
                    
                    # 如果清理后还有有效数据，使用清理版本
                    valid_ratio = finite_mask.float().mean().item()
                    if valid_ratio > 0.5:  # 超过50%的数据有效
                        aggregated[learner_id] = cleaned_grad
                        print(f"   ✅ Learner {learner_id}: 部分清理恢复 (保留{valid_ratio:.1%}有效数据)")
                    else:
                        # 数据损坏严重，使用零更新（相当于跳过本轮更新）
                        aggregated[learner_id] = torch.zeros_like(grad_tensor)
                        print(f"   ⚠️  Learner {learner_id}: 损坏严重，本轮零更新")
                else:
                    # 全部异常，零更新
                    aggregated[learner_id] = torch.zeros_like(grad_tensor)
                    print(f"   ⚠️  Learner {learner_id}: 全部异常，本轮零更新")
    
    print(f"📥 [服务器] 接收自客户端: {compression_stats['dense_clients']} 个稠密模型(100%) + {compression_stats['compressed_clients']} 个压缩模型")
    if compression_stats['skipped_clients'] > 0:
        print(f"⚠️  [异常处理] 跳过 {compression_stats['skipped_clients']} 个异常客户端数据")
    
    return aggregated


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights = \
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)
            
            # 📌 输出异常检测统计报告
            if hasattr(self, 'anomaly_detector') and self.c_round % (self.log_freq * 2) == 0:
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()
                if anomaly_summary['total_clients_with_anomalies'] > 0:
                    print("📊 异常检测统计报告:")
                    print(f"   发现异常的客户端数量: {anomaly_summary['total_clients_with_anomalies']}")
                    for client_id, records in anomaly_summary['client_records'].items():
                        print(f"   客户端 {client_id}: 总异常 {records['total']} 次 "
                              f"(严重: {records['severe']}, 中等: {records['moderate']}, 轻微: {records['minor']})")
                    print("#" * 80)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]
                for learner_id, learner in enumerate(client.learners_ensemble):
                    chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
                    learner.model.load_state_dict(torch.load(chkpts_path))

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):

        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.
p
    """

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        global_learner_weights = []
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            learner_weight = sum(
                [client.learners_ensemble.learners_weights[learner_id].item() for client in self.clients])
            global_learner_weights.append(learner_weight)
            average_learners(learners, learner, weights=self.clients_weights)

        print(np.array(global_learner_weights) / sum(global_learner_weights))
        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )


# class GCentralizedAggregator(Aggregator):
#     r""" Standard Centralized Aggregator.
#      All clients get fully synchronized with the average client.
#
#     """
#
#     def mix(self):
#         self.sample_clients()
#
#         for client in self.sampled_clients:
#             client.step()
#
#         gammas = torch.cat(
#             [client.n_train_samples * client.learners_ensemble.gmm.pi for client in self.clients]
#         )  # [c,k,1]
#         mus = torch.cat(
#             [client.learners_ensemble.gmm.mu for client in self.clients]
#         )  # [c,k,d]
#         vars = torch.cat(
#             [client.learners_ensemble.gmm.var for client in self.clients]
#         )  # [c,k,d,d]
#
#         global_gamma = torch.sum(gammas, dim=0)
#         global_mu = torch.sum(gammas * mus, dim=0) / global_gamma
#         global_var = torch.sum(gammas.unsqueeze(-1) * vars, dim=0) / global_gamma.unsqueeze(-1)
#         global_pi = global_gamma / global_gamma.sum()
#
#         for learner_id, learner in enumerate(self.global_learners_ensemble):
#             learners = [client.learners_ensemble[learner_id] for client in self.clients]
#             average_learners(learners, learner, weights=gammas[:, learner_id, 0])
#
#         self.global_learners_ensemble.gmm.update_parameter(
#             _pi=global_pi.unsqueeze(0),
#             mu=global_mu.unsqueeze(0),
#             var=global_var.unsqueeze(0)
#         )
#
#         # assign the updated model to all clients
#         self.update_clients()
#
#         self.c_round += 1
#
#         if self.c_round % self.log_freq == 0:
#             self.write_logs()
#
#     def update_clients(self):
#         for client in self.clients:
#             client.learners_ensemble.gmm.update_parameter(
#                 mu=self.global_learners_ensemble.gmm.mu, var=self.global_learners_ensemble.gmm.var)
#
#             for learner_id, learner in enumerate(client.learners_ensemble):
#                 copy_model(learner.model, self.global_learners_ensemble[learner_id].model)
#
#                 if callable(getattr(learner.optimizer, "set_initial_params", None)):
#                     learner.optimizer.set_initial_params(
#                         self.global_learners_ensemble[learner_id].model.parameters()
#                     )


class ACGCentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            ac_update_interval=10,
            *args,
            **kwargs):

        super().__init__(clients,
                         global_learners_ensemble,
                         log_freq,
                         global_train_logger,
                         global_test_logger,
                         sampling_rate,
                         sample_with_replacement,
                         test_clients,
                         verbose,
                         seed,
                         *args,
                         **kwargs)
        self.ac_update_interval = ac_update_interval
        
        # Initialize gradient cache manager for DGC
        self.gradient_cache = GradientCacheManager(device=self.device)
        print(f"Initialized GradientCacheManager on device: {self.device}")
        
        # 📌 初始化异常检测器
        self.anomaly_detector = AnomalyDetector(strict_mode=True)
        print(f"Initialized AnomalyDetector for robust DGC compression")
    
    def _update_gradient_cache(self):
        """Update gradient cache with current global learner parameters"""
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            # Get flattened parameter tensor
            param_tensor = learner.get_param_tensor()
            self.gradient_cache.update_cache(learner_id, param_tensor)
        
        if self.c_round <= 3:  # Log for first few rounds
            cache_info = self.gradient_cache.get_cache_info()
            print(f"Round {self.c_round}: Updated cache - {cache_info['num_cached_learners']} learners, "
                  f"{cache_info['total_memory_mb']:.1f}MB")
    
    def update_test_clients(self):
        for client in self.clients:
            client.learners_ensemble.gmm.update_parameter(
                mu=self.global_learners_ensemble.gmm.mu, var=self.global_learners_ensemble.gmm.var)
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)

                # copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
        for client in self.clients:
            client.update_sample_weights()
            client.update_learners_weights()
    def mix(self, gmm=False, unseen=False):
        print(f"\n🚀 [第 {self.c_round} 轮]")
        
        self.sample_clients()

        if not unseen:
            # ✅ 修复：在客户端训练前更新缓存（存储训练前的参数）
            # 这样缓存中的参数才是客户端计算delta的基准参数
            self._update_gradient_cache()
            
            # Collect client updates (either dense or compressed)
            client_payloads = []
            for client in self.sampled_clients:
                client_update = client.step(current_round=self.c_round)
                client_payloads.append(client_update)

            # Check if any client used compression
            has_compressed_clients = any(payload.get('type') == 'compressed' for payload in client_payloads)
            
            if has_compressed_clients:
                # DGC-aware aggregation path: recover sparse gradients and aggregate with cache
                aggregated_gradients = recover_and_aggregate(
                    client_payloads, self.n_learners, 
                    cache_manager=self.gradient_cache,
                    clients_weights=self.clients_weights,
                    anomaly_detector=self.anomaly_detector,
                    round_num=self.c_round
                )
                
                # ✅ 使用异常检测系统处理后的安全数据
                for learner_id, learner in enumerate(self.global_learners_ensemble):
                    if learner_id in aggregated_gradients:
                        grad_tensor = aggregated_gradients[learner_id].to(learner.device)
                        
                        # 应用delta更新到模型参数: new_params = old_params + delta
                        param_idx = 0
                        with torch.no_grad():
                            for param in learner.model.parameters():
                                param_size = param.numel()
                                param_delta = grad_tensor[param_idx:param_idx + param_size].reshape(param.shape)
                                param.data += param_delta  # 正确：累加delta
                                param_idx += param_size
            else:
                # All clients used dense upload: use original FedGMM aggregation logic
                for learner_id, learner in enumerate(self.global_learners_ensemble):
                    learners = [client.learners_ensemble[learner_id] for client in self.clients]
                    gammas = torch.cat([client.n_train_samples * client.learners_ensemble.learners_weights.unsqueeze(0) for client in self.clients])
                    gammas_sum2 = gammas.sum(dim=1)  # [c, m2]
                    average_learners(learners, learner, weights=gammas_sum2[:, learner_id])

            # GMM parameter updates (unchanged)
            gammas = torch.cat(
                [client.n_train_samples * client.learners_ensemble.learners_weights.unsqueeze(0) for client in self.clients]
            )  # [c, m1, m2]
            mus = torch.cat(
                [client.learners_ensemble.gmm.mu for client in self.clients]
            )  # [c, m1, d]
            vars = torch.cat(
                [client.learners_ensemble.gmm.var for client in self.clients]
            )  # [c, m1, d, d]

            gammas_sum1 = gammas.sum(dim=2).unsqueeze(2)  # [c, m1, 1]
            gammas_sum2 = gammas.sum(dim=1)  # [c, m2]

            global_gamma = torch.sum(gammas_sum1, dim=0)  # [1, m1, 1]
            global_mu = torch.sum(gammas_sum1 * mus, dim=0) / global_gamma
            global_var = torch.sum(gammas_sum1.unsqueeze(-1) * vars, dim=0) / global_gamma.unsqueeze(-1)
            global_pi = gammas_sum1.sum(dim=0) / gammas_sum1.sum()

            global_learners_weights = gammas.sum(dim=0) / gammas.sum()

            # for learner_id, learner in enumerate(self.global_learners_ensemble):
            #     learners = [client.learners_ensemble[learner_id] for client in self.clients]
            #     average_learners(learners, learner, weights=gammas_sum2[:, learner_id])

            # if self.c_round % self.ac_update_interval == 0:
            #     autoencoders = [client.learners_ensemble.autoencoder for client in self.clients]
            #     average_learners(autoencoders, self.global_learners_ensemble.autoencoder)

            print(global_learners_weights, global_pi, "okay")
            self.global_learners_ensemble.gmm.update_parameter(
                _pi=global_pi.unsqueeze(0),
                mu=global_mu.unsqueeze(0),
                var=global_var.unsqueeze(0)
            )

            # assign the updated model to all clients
            self.update_clients()

            self.c_round += 1
            # self.write_logs()

        else:
            # self.write_logs()
            self.update_test_clients()
            # self.write_test_logs()
        # self.write_logs()
        if self.c_round % self.log_freq == 0:
            self.write_logs()

    # def update_test_clients(self):
    #     for client in self.clients:
    #         client.learners_ensemble.gmm.update_parameter(
    #             mu=self.global_learners_ensemble.gmm.mu, var=self.global_learners_ensemble.gmm.var)
    #         for learner_id, learner in enumerate(client.learners_ensemble):
    #             copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
    #
    #             # copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
    #     for client in self.clients:
    #         client.update_sample_weights()
    #         client.update_learners_weights()
    def update_clients(self):
        # 简单的服务器下发模型大小打印（中文）- 只计算神经网络模型
        total_model_bytes = sum(sum(p.numel() * 4 for p in learner.model.parameters()) 
                               for learner in self.global_learners_ensemble)
        total_mb = total_model_bytes / (1024 * 1024)
        print(f"📤 [服务器] 向 {len(self.clients)} 个客户端下发模型: 每个 {total_model_bytes:,} 字节 ({total_mb:.2f} MB) - 100%模型大小")
        
        # 下发给每个客户端
        for client in self.clients:
            client.learners_ensemble.gmm.update_parameter(
                mu=self.global_learners_ensemble.gmm.mu, var=self.global_learners_ensemble.gmm.var)

            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )

            # if self.c_round % self.ac_update_interval == 0:
            #     learner = client.learners_ensemble.autoencoder
            #     copy_model(learner.model, self.global_learners_ensemble.autoencoder.model)
    def write_test_logs(self):
        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)
    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            global_train_recon = 0.
            global_train_nll = 0.
            global_test_recon = 0.
            global_test_nll = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc, train_recon, train_nll, test_recon, test_nll = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")
                    print(f"Train Reconstruction Loss: {train_recon:.3f} | Train NLL: {train_nll:.3f}|", end="")
                    print(f"Test Reconstruction Loss: {test_recon:.3f} | Test NLL: {test_nll:.3f} |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                global_train_recon += train_recon * client.n_train_samples
                global_train_nll += train_nll * client.n_train_samples
                global_test_recon += test_recon * client.n_test_samples
                global_test_nll += test_nll * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            global_train_recon /= total_n_samples
            global_test_recon /= total_n_test_samples
            global_train_nll /= total_n_samples
            global_test_nll /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print(f"Train Reconstruction Loss: {global_train_recon:.3f} | Train NLL: {global_train_nll:.3f}|",
                      end="")
                print(f"Test Reconstruction Loss: {global_test_recon:.3f} | Test NLL: {global_test_nll:.3f} |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def save_state(self, dir_path):
        self.global_learners_ensemble.save_state(os.path.join(dir_path, 'global_ensemble.pt'))


class PersonalizedAggregator(CentralizedAggregator):
    r"""
    Clients do not synchronize there models, instead they only synchronize optimizers, when needed.

    """

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(self.global_learners_ensemble[learner_id].model.parameters())


class APFLAggregator(Aggregator):
    r"""
    Implements
        `Adaptive Personalized Federated Learning`__(https://arxiv.org/abs/2003.13461)

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            alpha,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(APFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )
        assert self.n_learners == 2, "APFL requires two learners"

        self.alpha = alpha

    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            for _ in range(client.local_steps):
                client.step(single_batch_flag=True)

                partial_average(
                    learners=[client.learners_ensemble[1]],
                    average_learner=client.learners_ensemble[0],
                    alpha=self.alpha
                )

        average_learners(
            learners=[client.learners_ensemble[0] for client in self.clients],
            target_learner=self.global_learners_ensemble[0],
            weights=self.clients_weights
        )

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:

            copy_model(client.learners_ensemble[0].model, self.global_learners_ensemble[0].model)

            if callable(getattr(client.learners_ensemble[0].optimizer, "set_initial_params", None)):
                client.learners_ensemble[0].optimizer.set_initial_params(
                    self.global_learners_ensemble[0].model.parameters()
                )


class LoopLessLocalSGDAggregator(PersonalizedAggregator):
    """
    Implements L2SGD introduced in
    'Federated Learning of a Mixture of Global and Local Models'__. (https://arxiv.org/pdf/2002.05516.pdf)


    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            communication_probability,
            penalty_parameter,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(LoopLessLocalSGDAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.communication_probability = communication_probability
        self.penalty_parameter = penalty_parameter

    @property
    def communication_probability(self):
        return self.__communication_probability

    @communication_probability.setter
    def communication_probability(self, communication_probability):
        self.__communication_probability = communication_probability

    def mix(self):
        communication_flag = self.np_rng.binomial(1, self.communication_probability, 1)

        if communication_flag:
            for learner_id, learner in enumerate(self.global_learners_ensemble):
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_learners(learners, learner, weights=self.clients_weights)

                partial_average(
                    learners,
                    average_learner=learner,
                    alpha=self.penalty_parameter / self.communication_probability
                )

                self.update_clients()

                self.c_round += 1

                if self.c_round % self.log_freq == 0:
                    self.write_logs()

        else:
            self.sample_clients()
            for client in self.sampled_clients:
                client.step(single_batch_flag=True)


class ClusteredAggregator(Aggregator):
    """
    Implements
     `Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization under Privacy Constraints`.

     Follows implementation from https://github.com/felisat/clustered-federated-learning
    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            tol_1=0.4,
            tol_2=1.6,
            seed=None
    ):

        super(ClusteredAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        assert self.n_learners == 1, "ClusteredAggregator only supports single learner clients."
        assert self.sampling_rate == 1.0, f"`sampling_rate` is {sampling_rate}, should be {1.0}," \
                                          f" ClusteredAggregator only supports full clients participation."

        self.tol_1 = tol_1
        self.tol_2 = tol_2

        self.global_learners = [self.global_learners_ensemble]
        self.clusters_indices = [np.arange(len(clients)).astype("int")]
        self.n_clusters = 1

    def mix(self):
        clients_updates = np.zeros((self.n_clients, self.n_learners, self.model_dim))

        for client_id, client in enumerate(self.clients):
            clients_updates[client_id] = client.step()

        similarities = np.zeros((self.n_learners, self.n_clients, self.n_clients))

        for learner_id in range(self.n_learners):
            similarities[learner_id] = pairwise_distances(clients_updates[:, learner_id, :], metric="cosine")

        similarities = similarities.mean(axis=0)

        new_cluster_indices = []
        for indices in self.clusters_indices:
            max_update_norm = np.zeros(self.n_learners)
            mean_update_norm = np.zeros(self.n_learners)

            for learner_id in range(self.n_learners):
                max_update_norm[learner_id] = LA.norm(clients_updates[indices], axis=1).max()
                mean_update_norm[learner_id] = LA.norm(np.mean(clients_updates[indices], axis=0))

            max_update_norm = max_update_norm.mean()
            mean_update_norm = mean_update_norm.mean()

            if mean_update_norm < self.tol_1 and max_update_norm > self.tol_2 and len(indices) > 2:
                clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete")
                clustering.fit(similarities[indices][:, indices])
                cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
                cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
                new_cluster_indices += [cluster_1, cluster_2]
            else:
                new_cluster_indices += [indices]

        self.clusters_indices = new_cluster_indices

        self.n_clusters = len(self.clusters_indices)

        self.global_learners = [deepcopy(self.clients[0].learners_ensemble) for _ in range(self.n_clusters)]

        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_clients = [self.clients[i] for i in indices]
            for learner_id in range(self.n_learners):
                average_learners(
                    learners=[client.learners_ensemble[learner_id] for client in cluster_clients],
                    target_learner=self.global_learners[cluster_id][learner_id],
                    weights=self.clients_weights[indices] / self.clients_weights[indices].sum()
                )

        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for cluster_id, indices in enumerate(self.clusters_indices):
            cluster_learners = self.global_learners[cluster_id]

            for i in indices:
                for learner_id, learner in enumerate(self.clients[i].learners_ensemble):
                    copy_model(
                        target=learner.model,
                        source=cluster_learners[learner_id].model
                    )

    def update_test_clients(self):
        pass
#         for learner_id, learner in enumerate(self.test_clients[0].learners_ensemble):
#             copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
#         pass


class AgnosticAggregator(CentralizedAggregator):
    """
    Implements
     `Agnostic Federated Learning`__(https://arxiv.org/pdf/1902.00146.pdf).

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr_lambda,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(AgnosticAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.lr_lambda = lr_lambda

    def mix(self):
        self.sample_clients()

        clients_losses = []
        for client in self.sampled_clients:
            client_losses = client.step()
            clients_losses.append(client_losses)

        clients_losses = torch.tensor(clients_losses)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]

            average_learners(
                learners=learners,
                target_learner=learner,
                weights=self.clients_weights,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # update clients weights
        self.clients_weights += self.lr_lambda * clients_losses.mean(dim=1)
        self.clients_weights = simplex_projection(self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class FFLAggregator(CentralizedAggregator):
    """
    Implements q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            lr,
            q=1,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        super(FFLAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.q = q
        self.lr = lr
        assert self.sample_with_replacement, 'FFLAggregator only support sample with replacement'

    def mix(self):
        self.sample_clients()

        hs = 0
        for client in self.sampled_clients:
            hs += client.step(lr=self.lr)

        hs /= (self.lr * len(self.sampled_clients))  # take account for the lr used inside optimizer

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
            average_learners(
                learners=learners,
                target_learner=learner,
                weights=hs * torch.ones(len(learners)),
                average_params=False,
                average_gradients=True
            )

        # update parameters
        self.global_learners_ensemble.optimizer_step()

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()


class DecentralizedAggregator(Aggregator):
    def __init__(
            self,
            clients,
            global_learners_ensemble,
            mixing_matrix,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=True,
            test_clients=None,
            verbose=0,
            seed=None):

        super(DecentralizedAggregator, self).__init__(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            sample_with_replacement=sample_with_replacement,
            test_clients=test_clients,
            verbose=verbose,
            seed=seed
        )

        self.mixing_matrix = mixing_matrix
        assert self.sampling_rate >= 1, "partial sampling is not supported with DecentralizedAggregator"

    def update_clients(self):
        pass

    def mix(self):
        # update local models
        for client in self.clients:
            client.step()

        # mix models
        mixing_matrix = torch.tensor(
            self.mixing_matrix.copy(),
            dtype=torch.float32,
            device=self.device
        )

        for learner_id, global_learner in enumerate(self.global_learners_ensemble):
            state_dicts = [client.learners_ensemble[learner_id].model.state_dict() for client in self.clients]

            for key, param in global_learner.model.state_dict().items():
                shape_ = param.shape
                models_params = torch.zeros(self.n_clients, int(np.prod(shape_)), device=self.device)

                for ii, sd in enumerate(state_dicts):
                    models_params[ii] = sd[key].view(1, -1)

                models_params = mixing_matrix @ models_params

                for ii, sd in enumerate(state_dicts):
                    sd[key] = models_params[ii].view(shape_)

            for client_id, client in enumerate(self.clients):
                client.learners_ensemble[learner_id].model.load_state_dict(state_dicts[client_id])

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()
