"""
Enhanced RWKV Integration using External Repositories
Extends the existing echo_rwkv_bridge with external repository capabilities
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import importlib
import warnings

# Import the existing RWKV components
try:
    from .rwkv_repos import get_repo_manager, RWKVRepoType, RWKVRepoInfo
    from ..echo_rwkv_bridge import EchoRWKVIntegrationEngine
except ImportError:
    # Fallback imports for testing
    sys.path.append(str(Path(__file__).parent.parent))
    from integrations.rwkv_repos import get_repo_manager, RWKVRepoType, RWKVRepoInfo
    from echo_rwkv_bridge import EchoRWKVIntegrationEngine

logger = logging.getLogger(__name__)

class EnhancedRWKVBridge(EchoRWKVIntegrationEngine):
    """Enhanced RWKV Bridge using external BlinkDL repositories"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repo_manager = get_repo_manager()
        self.loaded_repos = {}
        self.external_models = {}
        self._initialize_external_integration()
    
    def _initialize_external_integration(self):
        """Initialize integration with external repositories"""
        logger.info("Initializing enhanced RWKV integration with external repositories")
        
        # Validate repository integrations
        validation_results = self.repo_manager.validate_integrations()
        available_repos = [name for name, valid in validation_results.items() if valid]
        
        logger.info(f"Available external repositories: {available_repos}")
        
        # Add successful repositories to Python path
        for repo_name in available_repos:
            try:
                self.repo_manager.add_to_python_path(repo_name)
                self.loaded_repos[repo_name] = self.repo_manager.get_repository(repo_name)
                logger.debug(f"Loaded repository: {repo_name}")
            except Exception as e:
                logger.warning(f"Failed to load repository {repo_name}: {e}")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from all repositories"""
        models = {}
        
        # Add external repository models
        for repo_name, repo_info in self.loaded_repos.items():
            models[f"external_{repo_name}"] = {
                "name": repo_name,
                "type": repo_info.repo_type.value,
                "description": repo_info.description,
                "path": str(repo_info.path),
                "entry_points": repo_info.entry_points,
                "source": "external_repository"
            }
        
        return models
    
    def load_external_model(self, repo_name: str, model_config: Optional[Dict] = None) -> bool:
        """Load a model from an external repository"""
        if repo_name not in self.loaded_repos:
            logger.error(f"Repository {repo_name} not available")
            return False
        
        repo_info = self.loaded_repos[repo_name]
        
        try:
            # Repository-specific loading logic
            if repo_info.repo_type == RWKVRepoType.MAIN_LM:
                return self._load_main_lm_model(repo_info, model_config)
            elif repo_info.repo_type == RWKVRepoType.CHAT:
                return self._load_chat_model(repo_info, model_config)
            elif repo_info.repo_type == RWKVRepoType.CUDA:
                return self._load_cuda_model(repo_info, model_config)
            elif repo_info.repo_type == RWKVRepoType.V2_PILE:
                return self._load_v2_pile_model(repo_info, model_config)
            elif repo_info.repo_type == RWKVRepoType.WORLD_MODEL:
                return self._load_world_model(repo_info, model_config)
            else:
                logger.warning(f"Unknown repository type: {repo_info.repo_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model from {repo_name}: {e}")
            return False
    
    def _load_main_lm_model(self, repo_info: RWKVRepoInfo, config: Optional[Dict]) -> bool:
        """Load model from RWKV-LM repository"""
        try:
            # Add the RWKV-LM src directory to Python path
            src_path = repo_info.path / "src"
            if src_path.exists():
                sys.path.insert(0, str(src_path))
            
            # Try to import and initialize RWKV model
            logger.info(f"Loading RWKV-LM from {repo_info.path}")
            
            # This is a placeholder - actual implementation would depend on RWKV-LM API
            self.external_models["main_lm"] = {
                "repo": repo_info,
                "config": config or {},
                "status": "loaded",
                "capabilities": ["generation", "training", "inference"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RWKV-LM model: {e}")
            return False
    
    def _load_chat_model(self, repo_info: RWKVRepoInfo, config: Optional[Dict]) -> bool:
        """Load model from ChatRWKV repository"""
        try:
            logger.info(f"Loading ChatRWKV from {repo_info.path}")
            
            # This is a placeholder - actual implementation would depend on ChatRWKV API
            self.external_models["chat"] = {
                "repo": repo_info,
                "config": config or {},
                "status": "loaded", 
                "capabilities": ["chat", "conversation", "web_interface"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ChatRWKV model: {e}")
            return False
    
    def _load_cuda_model(self, repo_info: RWKVRepoInfo, config: Optional[Dict]) -> bool:
        """Load CUDA accelerated model"""
        try:
            logger.info(f"Loading RWKV-CUDA from {repo_info.path}")
            
            # Check if CUDA is available
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, skipping CUDA model loading")
                    return False
            except ImportError:
                logger.warning("PyTorch not available, skipping CUDA model loading")
                return False
            
            self.external_models["cuda"] = {
                "repo": repo_info,
                "config": config or {},
                "status": "loaded",
                "capabilities": ["cuda_acceleration", "fast_inference"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RWKV-CUDA model: {e}")
            return False
    
    def _load_v2_pile_model(self, repo_info: RWKVRepoInfo, config: Optional[Dict]) -> bool:
        """Load RWKV-v2 Pile model"""
        try:
            logger.info(f"Loading RWKV-v2-RNN-Pile from {repo_info.path}")
            
            self.external_models["v2_pile"] = {
                "repo": repo_info,
                "config": config or {},
                "status": "loaded",
                "capabilities": ["pile_trained", "text_generation", "rnn_architecture"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RWKV-v2-RNN-Pile model: {e}")
            return False
    
    def _load_world_model(self, repo_info: RWKVRepoInfo, config: Optional[Dict]) -> bool:
        """Load WorldModel (Psychohistory) model"""
        try:
            logger.info(f"Loading WorldModel from {repo_info.path}")
            
            self.external_models["world_model"] = {
                "repo": repo_info,
                "config": config or {},
                "status": "loaded",
                "capabilities": ["psychohistory", "world_modeling", "llm_grounding"]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load WorldModel: {e}")
            return False
    
    def process_with_external_model(self, input_text: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """Process input using an external model"""
        if model_name not in self.external_models:
            logger.error(f"External model {model_name} not loaded")
            return {"error": f"Model {model_name} not available"}
        
        model_info = self.external_models[model_name]
        
        try:
            # Model-specific processing logic
            if model_name == "main_lm":
                return self._process_with_main_lm(input_text, model_info, **kwargs)
            elif model_name == "chat":
                return self._process_with_chat(input_text, model_info, **kwargs)
            elif model_name == "cuda":
                return self._process_with_cuda(input_text, model_info, **kwargs)
            elif model_name == "v2_pile":
                return self._process_with_v2_pile(input_text, model_info, **kwargs)
            elif model_name == "world_model":
                return self._process_with_world_model(input_text, model_info, **kwargs)
            else:
                return {"error": f"Unknown model type: {model_name}"}
                
        except Exception as e:
            logger.error(f"Error processing with {model_name}: {e}")
            return {"error": str(e)}
    
    def _process_with_main_lm(self, input_text: str, model_info: Dict, **kwargs) -> Dict[str, Any]:
        """Process with RWKV-LM model"""
        # Placeholder implementation
        return {
            "model": "RWKV-LM",
            "input": input_text,
            "output": f"[RWKV-LM] Processed: {input_text}",
            "type": "language_model_generation",
            "capabilities": model_info["capabilities"]
        }
    
    def _process_with_chat(self, input_text: str, model_info: Dict, **kwargs) -> Dict[str, Any]:
        """Process with ChatRWKV model"""
        # Placeholder implementation
        return {
            "model": "ChatRWKV",
            "input": input_text,
            "output": f"[ChatRWKV] Chat response: {input_text}",
            "type": "chat_response",
            "capabilities": model_info["capabilities"]
        }
    
    def _process_with_cuda(self, input_text: str, model_info: Dict, **kwargs) -> Dict[str, Any]:
        """Process with CUDA accelerated model"""
        # Placeholder implementation
        return {
            "model": "RWKV-CUDA",
            "input": input_text,
            "output": f"[CUDA-Accelerated] Fast processed: {input_text}",
            "type": "cuda_accelerated_generation",
            "capabilities": model_info["capabilities"]
        }
    
    def _process_with_v2_pile(self, input_text: str, model_info: Dict, **kwargs) -> Dict[str, Any]:
        """Process with RWKV-v2 Pile model"""
        # Placeholder implementation
        return {
            "model": "RWKV-v2-RNN-Pile",
            "input": input_text,
            "output": f"[Pile-Trained] Generated: {input_text}",
            "type": "pile_trained_generation",
            "capabilities": model_info["capabilities"]
        }
    
    def _process_with_world_model(self, input_text: str, model_info: Dict, **kwargs) -> Dict[str, Any]:
        """Process with WorldModel"""
        # Placeholder implementation
        return {
            "model": "WorldModel",
            "input": input_text,
            "output": f"[Psychohistory] World model analysis: {input_text}",
            "type": "world_modeling",
            "capabilities": model_info["capabilities"]
        }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of external repository integrations"""
        status = {
            "total_repositories": len(self.repo_manager.repositories),
            "loaded_repositories": len(self.loaded_repos),
            "external_models": len(self.external_models),
            "repositories": {},
            "models": {}
        }
        
        # Repository status
        for name, repo in self.repo_manager.repositories.items():
            status["repositories"][name] = {
                "available": repo.available,
                "loaded": name in self.loaded_repos,
                "path": str(repo.path) if repo.available else None,
                "type": repo.repo_type.value,
                "description": repo.description
            }
        
        # Model status
        for name, model in self.external_models.items():
            status["models"][name] = {
                "status": model["status"],
                "capabilities": model["capabilities"],
                "repository": model["repo"].name
            }
        
        return status

# Convenience function to create enhanced bridge
def create_enhanced_rwkv_bridge(**kwargs) -> EnhancedRWKVBridge:
    """Create an enhanced RWKV bridge with external repository support"""
    return EnhancedRWKVBridge(**kwargs)