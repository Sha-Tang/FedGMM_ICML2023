"""
Deep Gradient Compression (DGC) Compressor for FedGMM

Implements gradient compression with momentum accumulation for multiple learners.
Each learner maintains independent momentum cache for gradient compression.
"""

import torch
from typing import Dict, Tuple, Optional
from torch import Tensor


class DGCCompressor:
    """
    Deep Gradient Compression (DGC) for federated learning with multiple learners.
    
    Supports:
    - Multi-learner momentum caching (keyed by learner_id)
    - L2 gradient clipping
    - Momentum accumulation
    - Top-k sparsification
    - Momentum masking (clear momentum for non-uploaded positions)
    """
    
    def __init__(self, compress_ratio: float = 0.01, momentum: float = 0.9, clipping_norm: float = 1.0):
        """
        Initialize DGC Compressor.
        
        Args:
            compress_ratio: Compression ratio, e.g., 0.8 means keeping top 80% gradients
            momentum: Momentum coefficient for gradient accumulation
            clipping_norm: L2 norm threshold for gradient clipping (0 to disable)
        """
        self.compress_ratio = compress_ratio
        self.momentum = momentum
        self.clipping_norm = clipping_norm
        
        # Multi-learner momentum cache: Dict[learner_id, momentum_tensor]
        self.velocity_cache: Dict[int, Tensor] = {}
        
        # Statistics for logging
        self.total_elements = 0
        self.compressed_elements = 0
        
    def step(self, grad: torch.Tensor, learner_id: int) -> Tuple[Tensor, Tensor, Tuple[int]]:
        """
        Compress gradient for a specific learner.
        
        Args:
            grad: Current learner's gradient tensor (any shape, will be flattened)
            learner_id: Current learner's ID (int)
            
        Returns:
            indices: Selected top-k position indices (1D Tensor[int])
            values: Selected position values (1D Tensor[float])
            shape: Original grad shape (Tuple[int])
        """
        # 1. Record original shape and flatten gradient
        original_shape = grad.shape
        grad_flat = grad.flatten().clone()
        
        # 2. L2 gradient clipping (optional)
        if self.clipping_norm > 0:
            grad_norm = torch.norm(grad_flat)
            if grad_norm > self.clipping_norm:
                grad_flat = grad_flat * (self.clipping_norm / grad_norm)
        
        # 3. Initialize or retrieve momentum for this learner
        if learner_id not in self.velocity_cache:
            self.velocity_cache[learner_id] = torch.zeros_like(grad_flat)
        
        # 4. Momentum accumulation
        velocity = self.velocity_cache[learner_id]
        velocity = self.momentum * velocity + grad_flat
        
        # 5. Top-k selection based on absolute values
        total_elements = len(velocity)
        k = max(1, int(total_elements * self.compress_ratio))
        
        _, indices = torch.topk(torch.abs(velocity), k)
        indices = indices.sort()[0]  # Sort indices for better memory access
        
        # 6. Extract selected values
        values = velocity[indices]
        
        # 7. Momentum masking: clear momentum for non-uploaded positions
        mask = torch.zeros_like(velocity, dtype=torch.bool)
        mask[indices] = True
        velocity[~mask] = 0.0
        
        # 8. Update momentum cache
        self.velocity_cache[learner_id] = velocity
        
        # 9. Update statistics
        self.total_elements += total_elements
        self.compressed_elements += k
        
        return indices, values, original_shape
    
    def get_compression_stats(self) -> Dict[str, float]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression ratio and reduction statistics
        """
        if self.total_elements == 0:
            return {"compression_ratio": 0.0, "reduction_ratio": 0.0}
            
        actual_compression_ratio = self.compressed_elements / self.total_elements
        reduction_ratio = 1.0 - actual_compression_ratio
        
        return {
            "compression_ratio": actual_compression_ratio,
            "reduction_ratio": reduction_ratio,
            "total_elements": self.total_elements,
            "compressed_elements": self.compressed_elements
        }
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.total_elements = 0
        self.compressed_elements = 0
    
    def clear_cache(self, learner_id: Optional[int] = None):
        """
        Clear momentum cache.
        
        Args:
            learner_id: If specified, clear only this learner's cache.
                       If None, clear all caches.
        """
        if learner_id is not None:
            if learner_id in self.velocity_cache:
                del self.velocity_cache[learner_id]
        else:
            self.velocity_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of learners in cache."""
        return len(self.velocity_cache)
    
    def __repr__(self) -> str:
        return (f"DGCCompressor(compress_ratio={self.compress_ratio}, "
                f"momentum={self.momentum}, clipping_norm={self.clipping_norm}, "
                f"cached_learners={len(self.velocity_cache)})")


def reconstruct_dense_gradient(indices: Tensor, values: Tensor, 
                             original_shape: Tuple[int], device: torch.device) -> Tensor:
    """
    Reconstruct dense gradient from sparse representation.
    
    Args:
        indices: Sparse indices (1D tensor)
        values: Sparse values (1D tensor)
        original_shape: Original gradient shape
        device: Target device for reconstruction
        
    Returns:
        Dense gradient tensor with original shape
    """
    # Create zero tensor with original total size
    total_size = torch.prod(torch.tensor(original_shape)).item()
    dense_flat = torch.zeros(total_size, device=device, dtype=values.dtype)
    
    # Fill in sparse values
    dense_flat[indices] = values
    
    # Reshape to original shape
    return dense_flat.view(original_shape) 