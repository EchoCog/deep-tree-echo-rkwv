# RWKV-LM Integration Guide

This guide explains how to use the RWKV-LM integration in Deep Tree Echo.

## Overview

RWKV-LM (pronounced "RwaKuv") is an RNN with Transformer-level LLM performance that can be trained like a GPT transformer while maintaining linear time complexity and constant space requirements during inference.

## Features

### ✅ Implemented
- **RWKV-LM Repository Integration**: Full BlinkDL/RWKV-LM repository cloned and available
- **Python RWKV Package**: Installed and working with Deep Tree Echo
- **Cognitive Architecture Integration**: Memory, reasoning, and grammar membranes work with RWKV
- **Multiple RWKV Versions**: Support for RWKV v1 through v7 (including latest "Goose" v7)
- **Demo and Training Scripts**: Access to all RWKV-LM demo and training code
- **Repository Management**: Intelligent detection and management of RWKV repositories

### 🔄 Ready for Enhancement
- **Model Loading**: Download and load actual RWKV models for full functionality
- **Advanced Features**: Fine-tuning, custom training, and optimization
- **Additional Repositories**: ChatRWKV, RWKV-CUDA, WorldModel integrations

## Quick Start

### 1. Verify Installation

```python
from integrations.rwkv_repos import RWKVRepoManager
from simple_rwkv_integration import RWKV_AVAILABLE

# Check that RWKV package is available
print("RWKV Package Available:", RWKV_AVAILABLE)

# Check repository integration
repo_manager = RWKVRepoManager()
rwkv_lm = repo_manager.get_repository("RWKV-LM")
print("RWKV-LM Available:", rwkv_lm.available if rwkv_lm else False)
```

### 2. Use Cognitive Processing

```python
import asyncio
from echo_rwkv_bridge import EchoRWKVIntegrationEngine, CognitiveContext

async def process_with_rwkv():
    # Create integration engine
    engine = EchoRWKVIntegrationEngine(
        use_real_rwkv=True,
        use_cpp_backend=False
    )
    
    # Configure (without model for demo)
    config = {
        'rwkv': {
            'strategy': 'cpu fp32',
            'max_tokens': 100,
            'temperature': 0.8,
        },
        'enable_advanced_cognitive': True
    }
    
    # Initialize and process
    await engine.initialize(config)
    
    context = CognitiveContext(
        session_id="demo",
        user_input="What is RWKV?",
        conversation_history=[],
        memory_state={},
        processing_goals=["explain"],
        temporal_context=[],
        metadata={}
    )
    
    response = await engine.process_cognitive_input(context)
    print("Response:", response.integrated_output)

# Run the demo
asyncio.run(process_with_rwkv())
```

### 3. Explore RWKV-LM Repository

```python
from integrations.rwkv_repos import RWKVRepoManager

repo_manager = RWKVRepoManager()
rwkv_lm = repo_manager.get_repository("RWKV-LM")

if rwkv_lm and rwkv_lm.available:
    print("RWKV-LM Path:", rwkv_lm.path)
    print("Available Versions:", rwkv_lm.metadata.get('versions', []))
    print("Latest Version:", rwkv_lm.metadata.get('latest_version'))
    print("Architecture:", rwkv_lm.metadata.get('architecture'))
    print("Features:", rwkv_lm.metadata.get('features', []))
    
    # List demo files
    v7_path = rwkv_lm.path / "RWKV-v7"
    if v7_path.exists():
        demo_files = list(v7_path.glob("*demo*.py"))
        print("Demo Files:", [f.name for f in demo_files])
```

## Repository Structure

The RWKV-LM repository includes:

```
external/RWKV-LM/
├── RWKV-v1/          # RWKV version 1
├── RWKV-v2-RNN/      # RWKV version 2 (RNN mode)
├── RWKV-v3/          # RWKV version 3
├── RWKV-v4/          # RWKV version 4 (Pile models)
├── RWKV-v4neo/       # RWKV version 4 Neo
├── RWKV-v5/          # RWKV version 5 (World models)
├── RWKV-v6/          # RWKV version 6
├── RWKV-v7/          # RWKV version 7 "Goose" ⭐
│   ├── rwkv_v7_demo.py         # GPT mode demo
│   ├── rwkv_v7_demo_rnn.py     # RNN mode demo
│   ├── rwkv_v7_demo_fast.py    # Fast inference demo
│   ├── rwkv_v7_numpy.py        # NumPy implementation
│   └── train_temp/             # Training scripts
└── Research/         # Research papers and docs
```

## RWKV-7 "Goose" Features

RWKV-7 is the latest and most advanced version:

- **Linear Time Complexity**: O(n) instead of O(n²) like transformers
- **Constant Space**: No KV-cache required during inference
- **Attention-Free**: Uses recursive connections instead of attention
- **Meta-In-Context Learning**: Test-time training on context
- **Parallelizable Training**: Can be trained like GPT transformers
- **RNN Inference**: Can run in RNN mode for efficiency

## Using RWKV Models

### With Models (Enhanced Functionality)

To use actual RWKV models, download them from Hugging Face:

```python
# Example with a real model
config = {
    'rwkv': {
        'model_path': '/path/to/RWKV-4-Pile-169M-20220807-8023.pth',
        'strategy': 'cpu fp32',
        'max_tokens': 200,
        'temperature': 0.8,
        'top_p': 0.9,
    }
}
```

### Without Models (Mock Mode)

The integration works in mock mode when no model is provided:

```python
# Mock mode - uses simulated responses
config = {
    'rwkv': {
        'model_path': '',  # Empty path uses mock mode
        'strategy': 'cpu fp32',
    }
}
```

## Integration Testing

Run the comprehensive test suite:

```bash
# Test RWKV-LM integration
python test_rwkv_lm_integration.py

# Test simple RWKV integration
python test_simple_rwkv_integration.py

# Run integration demo
python demo_rwkv_lm_integration.py
```

## Development

### Adding New RWKV Repositories

To integrate additional RWKV repositories:

1. Clone the repository to `external/`
2. Update `src/integrations/rwkv_repos.py`
3. Add repository configuration
4. Test integration

### Model Integration

To add support for specific RWKV models:

1. Download model files (`.pth` or `.ggml` format)
2. Configure model path in integration
3. Test with actual inference
4. Optimize for your use case

## Next Steps

1. **Download Models**: Get RWKV models from [Hugging Face](https://huggingface.co/BlinkDL)
2. **Explore Training**: Use RWKV-v7 training scripts for custom models
3. **Optimize Performance**: Configure for your hardware (CPU/GPU)
4. **Integrate Features**: Add chat, API, or custom applications

## Resources

- **RWKV Website**: [rwkv.com](https://rwkv.com)
- **RWKV-LM Repository**: [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- **RWKV Discord**: [Join Community](https://discord.gg/bDSBUMeFpc)
- **Research Papers**: Available in `external/RWKV-LM/Research/`
- **Model Downloads**: [Hugging Face Models](https://huggingface.co/BlinkDL)

---

**🎉 RWKV-LM is now successfully integrated with Deep Tree Echo!**