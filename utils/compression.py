# utils/compression.py - 通信压缩工具模块

import torch
import numpy as np
from typing import Tuple, Optional


class CommunicationCompressor:
    """
    通信压缩器 - 实现Top-K压缩算法支持DGC残差补偿
    """
    
    def __init__(self, topk_ratio: float = 0.01, strategy: str = 'magnitude'):
        """
        初始化压缩器
        
        Args:
            topk_ratio: Top-K压缩比例 (0-1)
            strategy: 选择策略 ('magnitude' 或 'relative')
        """
        self.topk_ratio = topk_ratio
        self.strategy = strategy
        self.residual_cache = None
        self.original_size = 0
        self.compressed_size = 0
        
    def compress(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-K压缩算法
        
        Args:
            params: 要压缩的参数张量 (shape: [n_learners, model_dim])
            
        Returns:
            compressed_values: 压缩后的值
            indices: 选中的参数索引  
            shapes: 原始形状信息
        """
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params)
            
        original_shape = params.shape
        flat_params = params.flatten()
        
        # 计算Top-K数量
        total_params = flat_params.numel()
        k = max(1, int(total_params * self.topk_ratio))
        
        # 根据策略选择Top-K参数
        if self.strategy == 'magnitude':
            # 按绝对值大小选择
            _, indices = torch.topk(torch.abs(flat_params), k)
        elif self.strategy == 'relative':
            # 按相对变化选择 (需要当前参数值，这里简化为magnitude)
            _, indices = torch.topk(torch.abs(flat_params), k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        # 提取选中的值
        compressed_values = flat_params[indices]
        
        # 记录压缩统计
        self.original_size = total_params
        self.compressed_size = k
        
        return compressed_values, indices, torch.tensor(original_shape)
    
    def decompress(self, compressed_values: torch.Tensor, indices: torch.Tensor, 
                   shapes: torch.Tensor) -> torch.Tensor:
        """
        解压缩算法
        
        Args:
            compressed_values: 压缩的值
            indices: 参数索引
            shapes: 原始形状
            
        Returns:
            decompressed_params: 解压后的参数张量
        """
        # 重建完整参数张量
        total_params = shapes.prod().item()
        flat_params = torch.zeros(total_params, dtype=compressed_values.dtype, 
                                 device=compressed_values.device)
        flat_params[indices] = compressed_values
        
        # 恢复原始形状
        return flat_params.reshape(shapes.tolist())
    
    def update_residual(self, original: torch.Tensor, compressed: torch.Tensor):
        """
        更新残差缓存 (DGC算法)
        
        Args:
            original: 原始参数更新
            compressed: 压缩后参数更新
        """
        if self.residual_cache is None:
            self.residual_cache = torch.zeros_like(original)
            
        # 残差 = 原始更新 - 压缩更新
        self.residual_cache += original - compressed
    
    def get_residual_compensated_params(self, params: torch.Tensor) -> torch.Tensor:
        """
        获取残差补偿后的参数
        
        Args:
            params: 当前参数更新
            
        Returns:
            compensated_params: 补偿后的参数
        """
        if self.residual_cache is None:
            return params
            
        return params + self.residual_cache
    
    def reset_residual(self):
        """重置残差缓存"""
        if self.residual_cache is not None:
            self.residual_cache.zero_()
    
    def get_compression_ratio(self) -> float:
        """获取压缩比"""
        if self.original_size == 0:
            return 1.0
        return self.compressed_size / self.original_size
    
    def get_stats(self) -> dict:
        """获取压缩统计信息"""
        return {
            'original_size': self.original_size,
            'compressed_size': self.compressed_size, 
            'compression_ratio': self.get_compression_ratio(),
            'topk_ratio': self.topk_ratio,
            'strategy': self.strategy
        }


def create_compressor(args) -> Optional[CommunicationCompressor]:
    """
    根据参数创建压缩器实例
    
    Args:
        args: 包含压缩配置的参数对象
        
    Returns:
        压缩器实例或None
    """
    if not hasattr(args, 'use_dgc') or not args.use_dgc:
        return None
        
    return CommunicationCompressor(
        topk_ratio=args.topk_ratio,
        strategy=args.topk_strategy
    )


def should_compress(current_round: int, args) -> bool:
    """
    判断当前轮次是否应该压缩
    
    Args:
        current_round: 当前轮次
        args: 参数对象
        
    Returns:
        是否压缩
    """
    if not hasattr(args, 'use_dgc') or not args.use_dgc:
        return False
        
    # 预热期不压缩
    if current_round <= args.warmup_rounds:
        return False
        
    # 强制上传轮不压缩
    if current_round % args.force_upload_every == 0:
        return False
        
    return True


def should_reset_residual(current_round: int, args) -> bool:
    """
    判断是否应该重置残差缓存
    
    Args:
        current_round: 当前轮次
        args: 参数对象
        
    Returns:
        是否重置残差
    """
    if not hasattr(args, 'use_dgc') or not args.use_dgc:
        return False
        
    # 强制上传轮重置残差
    return current_round % args.force_upload_every == 0 