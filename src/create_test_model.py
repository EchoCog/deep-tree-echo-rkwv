"""
Local RWKV Model Simulator
Creates a minimal RWKV-compatible model file for testing purposes
"""

import torch
import json
import os
from pathlib import Path

def create_minimal_rwkv_model(output_path: str, model_size: str = "0.1B"):
    """Create a minimal RWKV model file for testing"""
    
    # Define model dimensions based on size
    size_configs = {
        "0.1B": {"n_layer": 6, "n_embd": 512},
        "0.4B": {"n_layer": 12, "n_embd": 768}, 
        "1.5B": {"n_layer": 24, "n_embd": 2048}
    }
    
    config = size_configs.get(model_size, size_configs["0.1B"])
    
    # Create minimal model state dict
    model_state = {}
    
    # Add basic RWKV components
    for layer in range(config["n_layer"]):
        layer_prefix = f"blocks.{layer}"
        
        # Time mixing components
        model_state[f"{layer_prefix}.att.time_mix_k"] = torch.randn(config["n_embd"])
        model_state[f"{layer_prefix}.att.time_mix_v"] = torch.randn(config["n_embd"])
        model_state[f"{layer_prefix}.att.time_mix_r"] = torch.randn(config["n_embd"])
        
        # Channel mixing components  
        model_state[f"{layer_prefix}.ffn.time_mix_k"] = torch.randn(config["n_embd"])
        model_state[f"{layer_prefix}.ffn.time_mix_r"] = torch.randn(config["n_embd"])
        
        # Layer norm
        model_state[f"{layer_prefix}.ln1.weight"] = torch.ones(config["n_embd"])
        model_state[f"{layer_prefix}.ln2.weight"] = torch.ones(config["n_embd"])
        
        # Linear layers
        model_state[f"{layer_prefix}.att.key.weight"] = torch.randn(config["n_embd"], config["n_embd"])
        model_state[f"{layer_prefix}.att.value.weight"] = torch.randn(config["n_embd"], config["n_embd"])
        model_state[f"{layer_prefix}.att.receptance.weight"] = torch.randn(config["n_embd"], config["n_embd"])
        
        model_state[f"{layer_prefix}.ffn.key.weight"] = torch.randn(config["n_embd"] * 4, config["n_embd"])
        model_state[f"{layer_prefix}.ffn.value.weight"] = torch.randn(config["n_embd"], config["n_embd"] * 4)
        model_state[f"{layer_prefix}.ffn.receptance.weight"] = torch.randn(config["n_embd"] * 4, config["n_embd"])
    
    # Embedding and output layers
    model_state["emb.weight"] = torch.randn(50277, config["n_embd"])  # RWKV vocab size
    model_state["head.weight"] = torch.randn(50277, config["n_embd"])
    model_state["ln_out.weight"] = torch.ones(config["n_embd"])
    
    # Add metadata
    model_state["_metadata"] = {
        "model_size": model_size,
        "n_layer": config["n_layer"],
        "n_embd": config["n_embd"],
        "vocab_size": 50277,
        "created_by": "rwkv_local_simulator",
        "version": "test_v1"
    }
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model_state, output_path)
    
    # Create info file
    info_path = output_path.replace('.pth', '_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            "model_name": f"RWKV-Test-{model_size}",
            "model_size": model_size,
            "parameters": config,
            "file_size_mb": os.path.getsize(output_path) // (1024 * 1024),
            "created_for_testing": True
        }, f, indent=2)
    
    print(f"Created test model: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) // (1024 * 1024)}MB")
    
    return output_path

if __name__ == "__main__":
    # Create test models
    models_dir = "/tmp/rwkv_models"
    
    # Create small test model
    create_minimal_rwkv_model(
        f"{models_dir}/RWKV-Test-0.1B.pth",
        "0.1B"
    )
    
    print("Test RWKV model created successfully!")