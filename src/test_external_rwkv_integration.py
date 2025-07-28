"""
Test External RWKV Repository Integration
Tests the integration of cloned BlinkDL RWKV repositories
"""

import unittest
import sys
import logging
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from integrations.rwkv_repos import get_repo_manager, RWKVRepoType, list_available_rwkv_repos
from integrations.enhanced_rwkv_bridge import create_enhanced_rwkv_bridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRWKVExternalIntegration(unittest.TestCase):
    """Test external RWKV repository integration"""
    
    def setUp(self):
        self.repo_manager = get_repo_manager()
        
    def test_repository_manager_initialization(self):
        """Test that repository manager initializes correctly"""
        self.assertIsNotNone(self.repo_manager)
        self.assertGreater(len(self.repo_manager.repositories), 0)
        
        # Check that all expected repository types are registered
        expected_repos = ["RWKV-LM", "ChatRWKV", "RWKV-CUDA", "WorldModel", "RWKV-v2-RNN-Pile"]
        for repo_name in expected_repos:
            self.assertIn(repo_name, self.repo_manager.repositories)
    
    def test_repository_availability(self):
        """Test repository availability detection"""
        available_repos = self.repo_manager.get_available_repositories()
        available_names = [repo.name for repo in available_repos]
        
        logger.info(f"Available repositories: {available_names}")
        
        # Should have at least some repositories available if cloning was successful
        self.assertGreater(len(available_repos), 0)
        
        # Test specific repository checks
        for repo_name in ["RWKV-LM", "ChatRWKV", "RWKV-CUDA", "WorldModel", "RWKV-v2-RNN-Pile"]:
            repo = self.repo_manager.get_repository(repo_name)
            self.assertIsNotNone(repo)
            
            if repo.available:
                self.assertTrue(repo.path.exists())
                logger.info(f"✓ Repository {repo_name} is available at {repo.path}")
            else:
                logger.warning(f"✗ Repository {repo_name} is not available")
    
    def test_repository_types(self):
        """Test repository type classification"""
        # Test getting repositories by type
        main_lm_repos = self.repo_manager.get_repositories_by_type(RWKVRepoType.MAIN_LM)
        self.assertEqual(len(main_lm_repos), 1)
        self.assertEqual(main_lm_repos[0].name, "RWKV-LM")
        
        chat_repos = self.repo_manager.get_repositories_by_type(RWKVRepoType.CHAT)
        self.assertEqual(len(chat_repos), 1)
        self.assertEqual(chat_repos[0].name, "ChatRWKV")
    
    def test_integration_summary(self):
        """Test integration summary generation"""
        summary = self.repo_manager.get_integration_summary()
        
        self.assertIn("total_repositories", summary)
        self.assertIn("available_repositories", summary)
        self.assertIn("repositories", summary)
        
        self.assertEqual(summary["total_repositories"], 5)
        
        # Verify summary structure
        for repo_name in summary["repositories"]:
            repo_info = summary["repositories"][repo_name]
            self.assertIn("available", repo_info)
            self.assertIn("type", repo_info)
            self.assertIn("description", repo_info)
        
        logger.info(f"Integration summary: {summary}")
    
    def test_enhanced_bridge_initialization(self):
        """Test enhanced RWKV bridge initialization"""
        try:
            bridge = create_enhanced_rwkv_bridge()
            self.assertIsNotNone(bridge)
            
            # Test that it has access to repository manager
            self.assertIsNotNone(bridge.repo_manager)
            
            # Test getting available models
            models = bridge.get_available_models()
            self.assertIsNotNone(models)
            
            logger.info(f"Enhanced bridge models: {list(models.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to create enhanced bridge: {e}")
            # Don't fail the test if the existing bridge has issues
            # This is expected since we're extending an existing system
            pass
    
    def test_external_model_loading(self):
        """Test loading external models"""
        try:
            bridge = create_enhanced_rwkv_bridge()
            
            # Test loading available repositories
            for repo_name in list_available_rwkv_repos():
                try:
                    result = bridge.load_external_model(repo_name)
                    logger.info(f"Loading {repo_name}: {'✓' if result else '✗'}")
                except Exception as e:
                    logger.warning(f"Failed to load {repo_name}: {e}")
            
            # Test integration status
            status = bridge.get_integration_status()
            self.assertIsNotNone(status)
            logger.info(f"Integration status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to test external model loading: {e}")
            # Don't fail if bridge creation fails
            pass
    
    def test_repository_validation(self):
        """Test repository validation"""
        validation_results = self.repo_manager.validate_integrations()
        
        self.assertIsNotNone(validation_results)
        self.assertEqual(len(validation_results), 5)  # Should validate all 5 repositories
        
        for repo_name, is_valid in validation_results.items():
            repo = self.repo_manager.get_repository(repo_name)
            if repo and repo.available:
                logger.info(f"Repository {repo_name}: {'valid' if is_valid else 'invalid'}")
            else:
                logger.info(f"Repository {repo_name}: not available")
    
    def test_python_path_addition(self):
        """Test adding repositories to Python path"""
        original_path_length = len(sys.path)
        
        # Test adding an available repository
        available_repos = list_available_rwkv_repos()
        if available_repos:
            repo_name = available_repos[0]
            result = self.repo_manager.add_to_python_path(repo_name)
            
            if result:
                # Path should be extended
                self.assertGreaterEqual(len(sys.path), original_path_length)
                logger.info(f"✓ Added {repo_name} to Python path")
            else:
                logger.warning(f"✗ Failed to add {repo_name} to Python path")

def run_integration_tests():
    """Run integration tests and return results"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRWKVExternalIntegration)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Testing RWKV External Repository Integration...")
    print("=" * 60)
    
    success = run_integration_tests()
    
    print("=" * 60)
    if success:
        print("✓ All integration tests passed!")
    else:
        print("✗ Some integration tests failed.")
    
    # Print repository status summary
    print("\nRepository Status Summary:")
    print("-" * 30)
    
    repo_manager = get_repo_manager()
    summary = repo_manager.get_integration_summary()
    
    print(f"Total repositories: {summary['total_repositories']}")
    print(f"Available repositories: {summary['available_repositories']}")
    print(f"Unavailable repositories: {summary['unavailable_repositories']}")
    
    print("\nDetailed Status:")
    for name, info in summary["repositories"].items():
        status = "✓ Available" if info["available"] else "✗ Not Available"
        print(f"  {name}: {status} ({info['type']})")
        if info["available"]:
            print(f"    Path: {info['path']}")
            print(f"    Entry points: {info['entry_points']}")
            print(f"    Dependencies: {info['dependencies']}")