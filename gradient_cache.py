"""
Gradient Cache Manager for DGC Compression in FedGMM

Manages historical gradient cache for completing missing positions in sparse gradients.
"""

import torch
from typing import Dict, Tuple, Optional
from torch import Tensor


class GradientCacheManager:
    """
    Gradient Cache Manager for DGC compression recovery.
    
    Caches full learner parameters from previous round's aggregation results,
    used to fill missing positions when clients upload sparse gradients.
    
    Cache Structure:
    - Key: learner_id (int) 
    - Value: torch.Tensor (dense model parameters as 1D tensor)
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize gradient cache manager.
        
        Args:
            device: Device to store cached tensors
        """
        self.device = device
        # Cache: {learner_id: full_parameter_tensor}
        self.cache: Dict[int, Tensor] = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def update_cache(self, learner_id: int, full_tensor: Tensor):
        """
        Update parameter cache for a specific learner.
        
        Called after global aggregation completes, before sending to clients.
        
        Args:
            learner_id: ID of the learner
            full_tensor: Complete parameter tensor (will be flattened and cached)
        """
        # Flatten and cache the tensor
        flat_tensor = full_tensor.flatten().clone().to(self.device)
        self.cache[learner_id] = flat_tensor
        
        if len(self.cache) <= 5:  # Avoid spam for many learners
            print(f"Cache updated for learner {learner_id}: {flat_tensor.shape} elements")
    
    def recover_from_compressed(
        self, 
        learner_id: int, 
        indices: Tensor, 
        values: Tensor, 
        shape: Tuple[int, ...]
    ) -> Tensor:
        """
        Recover complete dense delta from compressed sparse delta.
        
        核心逻辑：
        1. 客户端上传稀疏delta (只有top-k位置的增量)
        2. 未压缩位置的delta就是0 (表示这些位置没有显著变化)
        3. 恢复完整delta = 稀疏delta + 零填充
        
        Args:
            learner_id: ID of the learner
            indices: Positions of uploaded values (1D tensor)
            values: Uploaded delta values at sparse positions
            shape: Original parameter shape
            
        Returns:
            Complete dense delta tensor (1D, flattened)
        """
        total_elements = int(torch.prod(torch.tensor(shape)))
        
        # 初始化为零delta（未压缩位置的delta为0）
        delta_tensor = torch.zeros(total_elements, dtype=torch.float32, device=self.device)
        
        # 填入压缩位置的delta值
        if len(indices) > 0:
            # Ensure indices and values are on correct device
            indices = indices.to(self.device)
            values = values.to(self.device)
            
            # 设置压缩位置的delta值
            delta_tensor[indices] = values
        
        # 更新统计信息
        if learner_id in self.cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        return delta_tensor
    
    def has_cache(self, learner_id: int) -> bool:
        """Check if cache exists for a learner."""
        return learner_id in self.cache
    
    def clear_cache(self, learner_id: Optional[int] = None):
        """
        Clear cache for specific learner or all learners.
        
        Args:
            learner_id: If specified, clear only this learner's cache.
                       If None, clear all caches.
        """
        if learner_id is not None:
            if learner_id in self.cache:
                del self.cache[learner_id]
                print(f"Cache cleared for learner {learner_id}")
        else:
            self.cache.clear()
            print("All caches cleared")
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics and information."""
        total_elements = sum(tensor.numel() for tensor in self.cache.values())
        total_memory_mb = sum(tensor.element_size() * tensor.numel() for tensor in self.cache.values()) / (1024 * 1024)
        
        return {
            'cached_learners': list(self.cache.keys()),
            'num_cached_learners': len(self.cache),
            'total_cached_elements': total_elements,
            'total_memory_mb': total_memory_mb,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __repr__(self) -> str:
        info = self.get_cache_info()
        return (f"GradientCacheManager(learners={info['num_cached_learners']}, "
                f"memory={info['total_memory_mb']:.2f}MB, "
                f"hit_rate={info['hit_rate']:.2%})")


def test_gradient_cache_manager():
    """Test function for GradientCacheManager"""
    print("=== Testing GradientCacheManager ===")
    
    cache_manager = GradientCacheManager()
    
    # Test 1: Update cache
    learner_id = 0
    full_params = torch.randn(1000)
    cache_manager.update_cache(learner_id, full_params)
    
    # Test 2: Simulate compressed upload
    k = 100  # Top 10%
    indices = torch.randperm(1000)[:k]
    values = torch.randn(k)
    shape = (1000,)
    
    # Recover full gradient
    recovered = cache_manager.recover_from_compressed(learner_id, indices, values, shape)
    
    print(f"Original shape: {full_params.shape}")
    print(f"Compressed: {k} elements ({k/1000:.1%})")
    print(f"Recovered shape: {recovered.shape}")
    
    # Verify uploaded positions are correctly set
    print(f"Uploaded positions match: {torch.allclose(recovered[indices], values)}")
    
    # Test 3: Cache statistics
    print(f"Cache info: {cache_manager.get_cache_info()}")
    
    print("✅ GradientCacheManager test completed!")


if __name__ == "__main__":
    test_gradient_cache_manager() 