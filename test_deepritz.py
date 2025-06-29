#!/usr/bin/env python3
"""
Quick test script for DeepRitz implementation
"""

import os
import sys
import torch

# Add the src/python directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

def test_deepritz_import():
    """Test if DeepRitz can be imported correctly"""
    try:
        from deepritz_solver import DeepRitzSolver, DeepRitzTrainer
        print("✓ DeepRitz modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import DeepRitz modules: {e}")
        return False

def test_deepritz_basic_functionality():
    """Test basic functionality of DeepRitz"""
    try:
        from deepritz_solver import DeepRitzSolver, DeepRitzTrainer
        
        # Parameters
        device = torch.device('cpu')  # Use CPU for testing
        sigma_h = 9.5298e-4
        a = 18.515
        fr = 0.0
        ft = 0.2383
        fd = 1.0
        
        # Create model
        print("Creating DeepRitz model...")
        model = DeepRitzSolver(
            device=device,
            sigma_h=sigma_h,
            a=a,
            fr=fr,
            ft=ft,
            fd=fd,
            layers=[3, 32, 32, 1]  # Smaller network for testing
        )
        
        print(f"✓ DeepRitz model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        print("Testing forward pass...")
        x = torch.randn(10, 1, requires_grad=True)
        y = torch.randn(10, 1, requires_grad=True)
        t = torch.randn(10, 1, requires_grad=True)
        
        output = model(x, y, t)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Test energy functional loss
        print("Testing energy functional loss...")
        energy_loss = model.get_energy_functional_loss(x, y, t)
        print(f"✓ Energy loss computed: {energy_loss.item():.6f}")
        
        # Test initial condition loss
        print("Testing initial condition loss...")
        ic_loss = model.get_initial_condition_loss(x, y)
        print(f"✓ Initial condition loss computed: {ic_loss.item():.6f}")
        
        # Test trainer creation
        print("Testing trainer creation...")
        trainer = DeepRitzTrainer(model, device)
        print("✓ DeepRitz trainer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ DeepRitz test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== DeepRitz Implementation Test ===\n")
    
    # Check PyTorch availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    # Test imports
    if not test_deepritz_import():
        return
    
    # Test functionality
    if not test_deepritz_basic_functionality():
        return
    
    print("\n=== All tests passed! ===")
    print("\nYou can now train a DeepRitz model with:")
    print("python -m src.python.main deepritz-train --model-name test_model")

if __name__ == "__main__":
    main()
