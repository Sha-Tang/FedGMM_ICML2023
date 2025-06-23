#!/usr/bin/env python3
"""
Test script for DGCCompressor functionality
"""

import torch
from utils.dgc_compressor import DGCCompressor, reconstruct_dense_gradient

def test_dgc_compressor():
    """Test DGCCompressor with multiple learners"""
    print("=== Testing DGCCompressor ===")
    
    # Initialize compressor
    compressor = DGCCompressor(
        compress_ratio=0.3,  # Keep top 30%
        momentum=0.9,
        clipping_norm=1.0
    )
    
    print(f"Initialized: {compressor}")
    
    # Test with multiple learners
    n_learners = 3
    grad_size = 1000
    device = torch.device('cpu')
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        
        for learner_id in range(n_learners):
            # Simulate gradient for this learner
            grad = torch.randn(grad_size) * (learner_id + 1)  # Different scales
            
            # Compress gradient
            indices, values, shape = compressor.step(grad, learner_id)
            
            # Reconstruct gradient
            reconstructed = reconstruct_dense_gradient(indices, values, shape, device)
            
            # Calculate compression stats
            compressed_size = len(indices)
            original_size = grad.numel()
            actual_ratio = compressed_size / original_size
            
            print(f"  Learner {learner_id}: {original_size} → {compressed_size} "
                  f"({actual_ratio:.1%}) | Shape: {shape}")
            
            # Verify reconstruction shape
            assert reconstructed.shape == grad.shape, f"Shape mismatch: {reconstructed.shape} vs {grad.shape}"
    
    # Print final statistics
    stats = compressor.get_compression_stats()
    print(f"\n=== Final Statistics ===")
    print(f"Overall compression ratio: {stats['compression_ratio']:.1%}")
    print(f"Reduction ratio: {stats['reduction_ratio']:.1%}")
    print(f"Total elements processed: {stats['total_elements']}")
    print(f"Compressed elements: {stats['compressed_elements']}")
    print(f"Cached learners: {compressor.get_cache_size()}")

def test_different_shapes():
    """Test with different gradient shapes"""
    print("\n=== Testing Different Shapes ===")
    
    compressor = DGCCompressor(compress_ratio=0.2)
    
    test_shapes = [
        (100,),           # 1D
        (10, 10),         # 2D
        (2, 5, 10),       # 3D
        (2, 3, 4, 5)      # 4D
    ]
    
    for i, shape in enumerate(test_shapes):
        grad = torch.randn(shape)
        indices, values, reconstructed_shape = compressor.step(grad, learner_id=i)
        
        reconstructed = reconstruct_dense_gradient(
            indices, values, reconstructed_shape, torch.device('cpu')
        )
        
        print(f"Shape {shape}: Original size {grad.numel()}, "
              f"Compressed to {len(indices)} elements, "
              f"Reconstructed shape {reconstructed.shape}")
        
        assert reconstructed.shape == grad.shape

if __name__ == "__main__":
    test_dgc_compressor()
    test_different_shapes()
    print("\n✅ All tests passed!") 