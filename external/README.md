# External RWKV Repositories

This directory contains cloned BlinkDL RWKV repositories for integration with the Deep Tree Echo cognitive architecture.

## Repository Structure

- `RWKV-LM/` - ✅ Main RWKV language model repository (includes RWKV-7 "Goose")
- `rwkv.cpp/` - ✅ High-performance C++ RWKV implementation
- `ChatRWKV/` - 🔄 Chat interface powered by RWKV (planned)
- `RWKV-CUDA/` - 🔄 CUDA accelerated version (planned)
- `WorldModel/` - 🔄 Psychohistory project for LLM grounding (planned)
- `RWKV-v2-RNN-Pile/` - 🔄 RWKV-v2 trained on The Pile (planned)

## Integration

Each repository is integrated through the cognitive architecture's integration framework located in `../src/integrations/`.

## RWKV-LM Integration

The RWKV-LM repository provides:
- **RWKV-7 "Goose"**: Latest linear-time, constant-space, attention-free RNN architecture
- **Multi-version Support**: RWKV v1 through v7 implementations
- **Training Code**: Reference implementations for training custom models
- **Research Materials**: Comprehensive documentation and benchmarks

### Available RWKV Versions

- **RWKV v7** (Latest): Meta-in-context learner with test-time training
- **RWKV v6**: Enhanced architecture with improved reasoning
- **RWKV v5**: World models with multilingual support
- **RWKV v4**: Pile models for general text generation
- **RWKV v3-v1**: Earlier versions for research and comparison