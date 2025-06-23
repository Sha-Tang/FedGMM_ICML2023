#!/usr/bin/env python3
"""
Test script for server-side DGC aggregation functionality
"""

import sys
import os
import torch
import numpy as np
from aggregator import recover_and_aggregate

def test_recover_and_aggregate():
    """Test the recover_and_aggregate function with mixed dense/compressed payloads"""
    print("=== Testing Server-side DGC Aggregation ===")
    
    n_learners = 3
    model_dim = 1000
    n_clients = 4
    
    # Create mock client payloads (mix of dense and compressed)
    client_payloads = []
    
    # Client 0: Dense upload (warmup or DGC disabled)
    dense_updates = np.random.randn(n_learners, model_dim) * 0.1
    client_payloads.append({
        'type': 'dense',
        'updates': dense_updates,
        'compressed': False
    })
    
    # Client 1: Compressed upload
    compressed_data = {}
    for learner_id in range(n_learners):
        # Simulate DGC compression: top 10% elements
        full_grad = torch.randn(model_dim) * 0.1
        k = int(model_dim * 0.1)
        _, indices = torch.topk(torch.abs(full_grad), k)
        values = full_grad[indices]
        
        compressed_data[learner_id] = {
            'indices': indices.numpy(),
            'values': values.numpy(),
            'shape': (model_dim,),
            'dense_bytes': model_dim * 4,
            'compressed_bytes': k * 2 * 4
        }
    
    client_payloads.append({
        'type': 'compressed',
        'learners_data': compressed_data,
        'compressed': True,
        'total_compressed_bytes': k * 2 * 4 * n_learners,
        'total_dense_bytes': model_dim * 4 * n_learners
    })
    
    # Client 2: Another compressed upload  
    compressed_data_2 = {}
    for learner_id in range(n_learners):
        full_grad = torch.randn(model_dim) * 0.05
        k = int(model_dim * 0.15)  # Different compression ratio
        _, indices = torch.topk(torch.abs(full_grad), k)
        values = full_grad[indices]
        
        compressed_data_2[learner_id] = {
            'indices': indices.numpy(),
            'values': values.numpy(),
            'shape': (model_dim,),
            'dense_bytes': model_dim * 4,
            'compressed_bytes': k * 2 * 4
        }
    
    client_payloads.append({
        'type': 'compressed',
        'learners_data': compressed_data_2,
        'compressed': True
    })
    
    # Client 3: Another dense upload
    dense_updates_2 = np.random.randn(n_learners, model_dim) * 0.08
    client_payloads.append({
        'type': 'dense',
        'updates': dense_updates_2,
        'compressed': False
    })
    
    print(f"Created {len(client_payloads)} client payloads:")
    for i, payload in enumerate(client_payloads):
        print(f"  Client {i}: {payload['type']}")
    
    # Test aggregation
    try:
        aggregated = recover_and_aggregate(client_payloads, n_learners)
        
        print(f"\n✓ Aggregation successful!")
        print(f"Number of learners aggregated: {len(aggregated)}")
        
        for learner_id in range(n_learners):
            if learner_id in aggregated:
                grad_tensor = aggregated[learner_id]
                print(f"  Learner {learner_id}: {grad_tensor.shape}, norm={torch.norm(grad_tensor):.4f}")
            else:
                print(f"  Learner {learner_id}: Missing!")
                
        # Verify shapes are correct
        for learner_id in range(n_learners):
            assert learner_id in aggregated, f"Missing aggregated gradient for learner {learner_id}"
            assert aggregated[learner_id].shape == (model_dim,), f"Wrong shape for learner {learner_id}"
        
        print("\n✅ All shape checks passed!")
        
    except Exception as e:
        print(f"\n❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_compression_recovery():
    """Test recovery of compressed gradients specifically"""
    print("\n=== Testing Compressed Gradient Recovery ===")
    
    # Create a known sparse gradient
    model_dim = 100
    original_grad = torch.zeros(model_dim)
    
    # Set specific positions to known values
    test_indices = torch.tensor([5, 23, 47, 89])
    test_values = torch.tensor([1.5, -2.3, 0.8, -1.1])
    original_grad[test_indices] = test_values
    
    # Create compressed payload
    compressed_payload = [{
        'type': 'compressed',
        'learners_data': {
            0: {
                'indices': test_indices.numpy(),
                'values': test_values.numpy(),
                'shape': (model_dim,)
            }
        }
    }]
    
    # Recover
    aggregated = recover_and_aggregate(compressed_payload, 1)
    recovered_grad = aggregated[0]
    
    # Verify recovery
    print(f"Original non-zero positions: {test_indices.tolist()}")
    print(f"Original values: {test_values.tolist()}")
    print(f"Recovered non-zero positions: {torch.nonzero(recovered_grad).flatten().tolist()}")
    print(f"Recovered values: {recovered_grad[test_indices].tolist()}")
    
    # Check exact match
    if torch.allclose(original_grad, recovered_grad, atol=1e-6):
        print("✅ Perfect recovery!")
        return True
    else:
        print("❌ Recovery mismatch!")
        return False

if __name__ == "__main__":
    print("Testing server-side DGC aggregation functionality...\n")
    
    success1 = test_recover_and_aggregate()
    success2 = test_compression_recovery()
    
    if success1 and success2:
        print("\n🎉 All server-side DGC tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1) 