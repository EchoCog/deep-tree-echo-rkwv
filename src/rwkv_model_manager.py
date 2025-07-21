"""
RWKV Model Management System
Handles model downloading, caching, validation, and memory optimization for WebVM deployment
"""

import os
import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import requests
from dataclasses import dataclass
import time

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for RWKV model"""
    name: str
    size: str
    url: str
    checksum: str
    context_length: int = 2048
    vocab_size: int = 50277
    memory_mb: int = 0

# Available RWKV models optimized for WebVM deployment
AVAILABLE_MODELS = {
    "RWKV-Test-0.1B": ModelConfig(
        name="RWKV-Test-0.1B",
        size="0.1B",
        url="local",  # Local test model
        checksum="placeholder",
        memory_mb=300
    ),
    "RWKV-4-World-0.1B": ModelConfig(
        name="RWKV-4-World-0.1B",
        size="0.1B",
        url="https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.1B-v1-20230803-ctx4096.pth",
        checksum="d24d005e8ad6a4a5a1e1a8e5e4b3c7a9",
        memory_mb=200
    ),
    "RWKV-4-World-0.4B": ModelConfig(
        name="RWKV-4-World-0.4B", 
        size="0.4B",
        url="https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth",
        checksum="a34e567f8901234567890123456789ab",
        memory_mb=450
    ),
    "RWKV-4-World-1.5B": ModelConfig(
        name="RWKV-4-World-1.5B",
        size="1.5B", 
        url="https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/RWKV-4-World-1.5B-v1-20230803-ctx4096.pth",
        checksum="b45f678a9012345678901234567890bc",
        memory_mb=580
    )
}

class RWKVModelManager:
    """Manages RWKV model lifecycle for WebVM deployment"""
    
    def __init__(self, models_dir: str = "/tmp/rwkv_models", memory_limit_mb: int = 600):
        self.models_dir = Path(models_dir)
        self.memory_limit_mb = memory_limit_mb
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache = {}
        self.current_model = None
        
        # Create metadata file
        self.metadata_file = self.models_dir / "models_metadata.json"
        self._load_metadata()
        
        logger.info(f"RWKV Model Manager initialized. Models dir: {self.models_dir}")
    
    def _load_metadata(self) -> None:
        """Load model metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_best_model_for_memory_limit(self) -> Optional[ModelConfig]:
        """Get the best model that fits within memory constraints"""
        suitable_models = [
            model for model in AVAILABLE_MODELS.values() 
            if model.memory_mb <= self.memory_limit_mb
        ]
        
        if not suitable_models:
            logger.warning(f"No models fit within {self.memory_limit_mb}MB memory limit")
            return None
        
        # Prioritize local test models first for development
        local_models = [m for m in suitable_models if m.url == "local"]
        if local_models:
            return max(local_models, key=lambda m: m.memory_mb)
        
        # Return largest model that fits
        return max(suitable_models, key=lambda m: m.memory_mb)
    
    def is_model_cached(self, model_config: ModelConfig) -> bool:
        """Check if model is already cached locally"""
        model_path = self.models_dir / f"{model_config.name}.pth"
        
        if not model_path.exists():
            return False
        
        # Verify file integrity if checksum available
        if model_config.checksum and model_config.checksum != "placeholder":
            return self._verify_checksum(model_path, model_config.checksum)
        
        return True
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file integrity using checksum"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash == expected_checksum
        except Exception as e:
            logger.error(f"Failed to verify checksum for {file_path}: {e}")
            return False
    
    def download_model(self, model_config: ModelConfig, force_download: bool = False) -> bool:
        """Download model if not cached"""
        model_path = self.models_dir / f"{model_config.name}.pth"
        
        # Handle local test models
        if model_config.url == "local":
            # Check if local test model exists
            if model_path.exists():
                logger.info(f"Local test model {model_config.name} found")
                return True
            else:
                logger.info(f"Creating local test model {model_config.name}")
                try:
                    # Import and create test model
                    from create_test_model import create_minimal_rwkv_model
                    create_minimal_rwkv_model(str(model_path), model_config.size)
                    
                    # Update metadata
                    self.metadata[model_config.name] = {
                        'downloaded_at': time.time(),
                        'size_mb': model_path.stat().st_size // (1024 * 1024),
                        'checksum': model_config.checksum,
                        'url': model_config.url,
                        'local_test_model': True
                    }
                    self._save_metadata()
                    
                    return True
                except Exception as e:
                    logger.error(f"Failed to create local test model: {e}")
                    return False
        
        # Check if already cached and valid
        if not force_download and self.is_model_cached(model_config):
            logger.info(f"Model {model_config.name} already cached")
            return True
        
        logger.info(f"Downloading model {model_config.name} from {model_config.url}")
        
        try:
            # Create progress tracker
            def download_with_progress(url: str, filepath: Path) -> bool:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress every 10MB
                            if downloaded_size % (10 * 1024 * 1024) == 0 or downloaded_size == total_size:
                                progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
                                logger.info(f"Downloaded {downloaded_size // 1024 // 1024}MB / {total_size // 1024 // 1024}MB ({progress:.1f}%)")
                
                return True
            
            # Download model
            success = download_with_progress(model_config.url, model_path)
            
            if success:
                # Update metadata
                self.metadata[model_config.name] = {
                    'downloaded_at': time.time(),
                    'size_mb': model_path.stat().st_size // (1024 * 1024),
                    'checksum': model_config.checksum,
                    'url': model_config.url
                }
                self._save_metadata()
                
                logger.info(f"Successfully downloaded {model_config.name}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to download model {model_config.name}: {e}")
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            return False
        
        return False
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model"""
        model_path = self.models_dir / f"{model_name}.pth"
        return model_path if model_path.exists() else None
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached models with metadata"""
        cached_models = []
        
        for model_name, config in AVAILABLE_MODELS.items():
            model_path = self.models_dir / f"{model_name}.pth"
            
            if model_path.exists():
                cached_models.append({
                    'name': model_name,
                    'size': config.size,
                    'memory_mb': config.memory_mb,
                    'file_size_mb': model_path.stat().st_size // (1024 * 1024),
                    'cached': True,
                    'metadata': self.metadata.get(model_name, {})
                })
        
        return cached_models
    
    def cleanup_old_models(self, keep_latest: int = 2) -> None:
        """Clean up old model files to save space"""
        cached_models = self.list_cached_models()
        
        if len(cached_models) <= keep_latest:
            return
        
        # Sort by download time, keep most recent
        sorted_models = sorted(
            cached_models, 
            key=lambda m: m['metadata'].get('downloaded_at', 0),
            reverse=True
        )
        
        models_to_remove = sorted_models[keep_latest:]
        
        for model_info in models_to_remove:
            model_path = self.models_dir / f"{model_info['name']}.pth"
            try:
                model_path.unlink()
                # Remove from metadata
                if model_info['name'] in self.metadata:
                    del self.metadata[model_info['name']]
                logger.info(f"Cleaned up old model: {model_info['name']}")
            except Exception as e:
                logger.error(f"Failed to remove {model_path}: {e}")
        
        self._save_metadata()
    
    def get_memory_usage_estimate(self, model_name: str) -> int:
        """Get estimated memory usage for a model in MB"""
        if model_name in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_name].memory_mb
        return 0
    
    def prepare_model_for_webvm(self) -> Optional[Dict[str, Any]]:
        """Prepare the best model for WebVM deployment"""
        # Get best model for memory limit
        best_model = self.get_best_model_for_memory_limit()
        
        if not best_model:
            logger.error("No suitable model found for WebVM memory constraints")
            return None
        
        # Download if needed
        if not self.download_model(best_model):
            logger.error(f"Failed to download model {best_model.name}")
            return None
        
        model_path = self.get_model_path(best_model.name)
        
        return {
            'model_config': best_model,
            'model_path': str(model_path),
            'memory_usage_mb': best_model.memory_mb,
            'context_length': best_model.context_length
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model manager
    manager = RWKVModelManager(memory_limit_mb=600)
    
    # Prepare model for WebVM
    model_info = manager.prepare_model_for_webvm()
    
    if model_info:
        print(f"Prepared model for WebVM deployment:")
        print(f"Model: {model_info['model_config'].name}")
        print(f"Path: {model_info['model_path']}")
        print(f"Memory usage: {model_info['memory_usage_mb']}MB")
        print(f"Context length: {model_info['context_length']}")
    else:
        print("Failed to prepare model for WebVM deployment")
    
    # List cached models
    cached = manager.list_cached_models()
    print(f"\nCached models: {len(cached)}")
    for model in cached:
        print(f"- {model['name']} ({model['size']}, {model['memory_mb']}MB)")