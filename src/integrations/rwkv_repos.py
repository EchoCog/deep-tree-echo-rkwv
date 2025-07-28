"""
RWKV External Repository Integration Framework
Manages integration with cloned BlinkDL RWKV repositories
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import json

logger = logging.getLogger(__name__)

class RWKVRepoType(Enum):
    """Types of RWKV repositories"""
    MAIN_LM = "RWKV-LM"           # Main RWKV language model
    CHAT = "ChatRWKV"             # Chat interface 
    CUDA = "RWKV-CUDA"           # CUDA accelerated version
    WORLD_MODEL = "WorldModel"    # Psychohistory project
    V2_PILE = "RWKV-v2-RNN-Pile" # RWKV-v2 trained on The Pile

@dataclass
class RWKVRepoInfo:
    """Information about an RWKV repository"""
    name: str
    repo_type: RWKVRepoType
    path: Path
    description: str
    available: bool = False
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    entry_points: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class RWKVRepoManager:
    """Manages BlinkDL RWKV repository integrations"""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent.parent / "external"
        self.repositories: Dict[str, RWKVRepoInfo] = {}
        self._initialize_repositories()
    
    def _initialize_repositories(self):
        """Initialize repository information"""
        repo_configs = [
            {
                "name": "RWKV-LM",
                "repo_type": RWKVRepoType.MAIN_LM,
                "description": "Main RWKV language model with RWKV-7 'Goose'",
                "entry_points": {
                    "train": "train.py",
                    "inference": "src/run.py",
                    "model": "src/model.py"
                }
            },
            {
                "name": "ChatRWKV", 
                "repo_type": RWKVRepoType.CHAT,
                "description": "ChatGPT-like interface powered by RWKV",
                "entry_points": {
                    "chat": "chat.py",
                    "api": "API_DEMO.py",
                    "web": "web.py"
                }
            },
            {
                "name": "RWKV-CUDA",
                "repo_type": RWKVRepoType.CUDA, 
                "description": "CUDA accelerated RWKV implementation",
                "entry_points": {
                    "cuda_kernel": "cuda/wkv_cuda_kernel.cu",
                    "ops": "wkv_op.py"
                }
            },
            {
                "name": "WorldModel",
                "repo_type": RWKVRepoType.WORLD_MODEL,
                "description": "Psychohistory project for LLM grounding", 
                "entry_points": {}
            },
            {
                "name": "RWKV-v2-RNN-Pile", 
                "repo_type": RWKVRepoType.V2_PILE,
                "description": "RWKV-v2 RNN trained on The Pile dataset",
                "entry_points": {
                    "model": "inference.py",
                    "chat": "chat.py"
                }
            }
        ]
        
        for config in repo_configs:
            repo_path = self.base_path / config["name"]
            repo_info = RWKVRepoInfo(
                name=config["name"],
                repo_type=config["repo_type"],
                path=repo_path,
                description=config["description"],
                available=repo_path.exists(),
                entry_points=config["entry_points"]
            )
            
            if repo_info.available:
                self._scan_repository(repo_info)
            
            self.repositories[config["name"]] = repo_info
            
        logger.info(f"Initialized {len(self.repositories)} RWKV repositories")
    
    def _scan_repository(self, repo_info: RWKVRepoInfo):
        """Scan repository for additional information"""
        try:
            # Check for README
            readme_files = list(repo_info.path.glob("README*"))
            if readme_files:
                repo_info.metadata["readme"] = str(readme_files[0])
            
            # Check for requirements.txt
            requirements_file = repo_info.path / "requirements.txt"
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    repo_info.dependencies = [
                        line.strip() for line in f.readlines() 
                        if line.strip() and not line.startswith('#')
                    ]
            
            # Check for setup.py or pyproject.toml
            if (repo_info.path / "setup.py").exists():
                repo_info.metadata["has_setup"] = True
            if (repo_info.path / "pyproject.toml").exists():
                repo_info.metadata["has_pyproject"] = True
            
            # Validate entry points exist
            valid_entry_points = {}
            for name, path in repo_info.entry_points.items():
                full_path = repo_info.path / path
                if full_path.exists():
                    valid_entry_points[name] = path
                    
            repo_info.entry_points = valid_entry_points
            
            logger.debug(f"Scanned repository {repo_info.name}: {len(valid_entry_points)} entry points found")
            
        except Exception as e:
            logger.warning(f"Error scanning repository {repo_info.name}: {e}")
    
    def get_repository(self, name: str) -> Optional[RWKVRepoInfo]:
        """Get repository info by name"""
        return self.repositories.get(name)
    
    def get_available_repositories(self) -> List[RWKVRepoInfo]:
        """Get list of available repositories"""
        return [repo for repo in self.repositories.values() if repo.available]
    
    def get_repositories_by_type(self, repo_type: RWKVRepoType) -> List[RWKVRepoInfo]:
        """Get repositories by type"""
        return [repo for repo in self.repositories.values() if repo.repo_type == repo_type]
    
    def add_to_python_path(self, repo_name: str) -> bool:
        """Add repository to Python path for importing"""
        repo = self.get_repository(repo_name)
        if not repo or not repo.available:
            logger.warning(f"Repository {repo_name} not available")
            return False
            
        repo_path_str = str(repo.path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
            logger.info(f"Added {repo_name} to Python path")
            return True
        return True
    
    def import_module_from_repo(self, repo_name: str, module_path: str, module_name: Optional[str] = None):
        """Import a module from a repository"""
        repo = self.get_repository(repo_name)
        if not repo or not repo.available:
            raise ImportError(f"Repository {repo_name} not available")
        
        full_path = repo.path / module_path
        if not full_path.exists():
            raise ImportError(f"Module {module_path} not found in {repo_name}")
        
        if module_name is None:
            module_name = full_path.stem
            
        spec = importlib.util.spec_from_file_location(module_name, full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module {module_path} from {repo_name}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of repository integrations"""
        available_count = len(self.get_available_repositories())
        total_count = len(self.repositories)
        
        summary = {
            "total_repositories": total_count,
            "available_repositories": available_count, 
            "unavailable_repositories": total_count - available_count,
            "repositories": {}
        }
        
        for name, repo in self.repositories.items():
            summary["repositories"][name] = {
                "available": repo.available,
                "type": repo.repo_type.value,
                "description": repo.description,
                "entry_points": len(repo.entry_points),
                "dependencies": len(repo.dependencies),
                "path": str(repo.path) if repo.available else None
            }
        
        return summary
    
    def validate_integrations(self) -> Dict[str, bool]:
        """Validate all repository integrations"""
        results = {}
        
        for name, repo in self.repositories.items():
            try:
                if not repo.available:
                    results[name] = False
                    continue
                
                # Basic validation - check if we can add to path
                self.add_to_python_path(name)
                
                # Check entry points exist
                valid_entry_points = 0
                for entry_name, entry_path in repo.entry_points.items():
                    if (repo.path / entry_path).exists():
                        valid_entry_points += 1
                
                results[name] = valid_entry_points > 0 or len(repo.entry_points) == 0
                
            except Exception as e:
                logger.error(f"Validation failed for {name}: {e}")
                results[name] = False
        
        return results

# Global repository manager instance
_repo_manager = None

def get_repo_manager() -> RWKVRepoManager:
    """Get global repository manager instance"""
    global _repo_manager
    if _repo_manager is None:
        _repo_manager = RWKVRepoManager()
    return _repo_manager

def get_rwkv_repository(name: str) -> Optional[RWKVRepoInfo]:
    """Convenience function to get repository info"""
    return get_repo_manager().get_repository(name)

def list_available_rwkv_repos() -> List[str]:
    """Convenience function to list available repository names"""
    return [repo.name for repo in get_repo_manager().get_available_repositories()]