#!/usr/bin/env python3
"""
Test script for DGC-enabled client functionality
"""

import sys
import os
import torch
import numpy as np
from utils.args import parse_args
from utils.dgc_compressor import DGCCompressor
from client import ACGMixtureClient

class MockLearner:
    """Mock learner for testing"""
    def __init__(self, model_dim=1000):
        self.model_dim = model_dim
        
    def get_param_tensor(self):
        return torch.randn(self.model_dim)

class MockLearnersEnsemble:
    """Mock learners ensemble for testing"""
    def __init__(self, n_learners=3, model_dim=1000):
        self.model_dim = model_dim
        self.learners = [MockLearner(model_dim) for _ in range(n_learners)]
        self.learners_weights = torch.ones(n_learners) / n_learners
        self.device = torch.device('cpu')
        
    def __len__(self):
        return len(self.learners)
        
    def fit_epochs(self, iterator, n_epochs, weights=None):
        """Simulate training and return mock gradients"""
        n_learners = len(self.learners)
        client_updates = np.random.randn(n_learners, self.model_dim)
        return client_updates
        
    def initialize_gmm(self, iterator):
        pass
        
    def calc_samples_weights(self, iterator):
        return torch.ones(3, 100) / 3  # Mock weights
        
    def m_step(self, sample_weights, iterator):
        pass
        
    def free_gradients(self):
        pass

class MockLogger:
    """Mock TensorBoard logger"""
    def add_scalar(self, tag, value, step):
        print(f"Log: {tag} = {value:.4f} at step {step}")

def test_dgc_client():
    """Test DGC-enabled client"""
    print("=== Testing DGC Client Functionality ===")
    
    # Create mock args
    args = parse_args(['cifar10', 'FedGMM', '--use_dgc', '--compress_ratio', '0.3', '--warmup_rounds', '2'])
    
    # Create mock components
    learners_ensemble = MockLearnersEnsemble(n_learners=3, model_dim=1000)
    logger = MockLogger()
    
    # Create DGC-enabled client
    client = ACGMixtureClient(
        learners_ensemble=learners_ensemble,
        train_iterator=None,  # Mock
        val_iterator=None,    # Mock  
        test_iterator=None,   # Mock
        logger=logger,
        local_steps=1,
        save_path="./test_save",
        tune_locally=False,
        client_id=0,
        args=args
    )
    
    print(f"Client initialized with DGC: {client.use_dgc}")
    print(f"Compression ratio: {client.compress_ratio}")
    print(f"Warmup rounds: {client.warmup_rounds}")
    
    # Test compression behavior over multiple rounds
    for round_num in range(5):
        print(f"\n--- Round {round_num} ---")
        
        # Simulate training step
        updates = client._compress_updates_for_upload(
            np.random.randn(3, 1000)  # Mock gradients for 3 learners
        )
        
        # Check compression status
        if round_num < client.warmup_rounds:
            assert updates['type'] == 'dense', f"Expected dense in warmup round {round_num}"
            print(f"✓ Warmup mode: Dense upload")
        else:
            assert updates['type'] == 'compressed', f"Expected compressed in round {round_num}"
            print(f"✓ Compression mode: Sparse upload")
            
            # Check compression stats
            total_compressed = updates['total_compressed_bytes']
            total_dense = updates['total_dense_bytes']
            reduction = 1.0 - (total_compressed / total_dense)
            print(f"  Compression: {total_dense} → {total_compressed} bytes ({reduction:.1%} reduction)")
        
        # Update round for client
        client.current_round = round_num

def test_compression_stats():
    """Test compression statistics logging"""
    print("\n=== Testing Compression Statistics ===")
    
    # Create DGC compressor
    compressor = DGCCompressor(compress_ratio=0.2, momentum=0.9)
    
    # Test compression over multiple steps
    for step in range(3):
        for learner_id in range(3):
            grad = torch.randn(1000) * (learner_id + 1)
            indices, values, shape = compressor.step(grad, learner_id)
            
            compression_ratio = len(indices) / grad.numel()
            print(f"Step {step}, Learner {learner_id}: {grad.numel()} → {len(indices)} ({compression_ratio:.1%})")
    
    # Print final stats
    stats = compressor.get_compression_stats()
    print(f"Final stats: {stats}")

if __name__ == "__main__":
    # Ensure imports work
    os.makedirs("./test_save", exist_ok=True)
    
    try:
        test_dgc_client()
        test_compression_stats()
        print("\n✅ All DGC client tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists("./test_save"):
            import shutil
            shutil.rmtree("./test_save") 