#!/usr/bin/env python3
"""
Test RWKV-LM Integration with Deep Tree Echo
Validates that the RWKV-LM repository is properly integrated and working
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from integrations.rwkv_repos import RWKVRepoManager, RWKVRepoType
from simple_rwkv_integration import RWKV_AVAILABLE, SimpleRWKVIntegration

class TestRWKVLMIntegration(unittest.TestCase):
    """Test RWKV-LM integration functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.repo_manager = RWKVRepoManager()
    
    def test_rwkv_package_available(self):
        """Test that RWKV package is available"""
        self.assertTrue(RWKV_AVAILABLE, "RWKV package should be available after installation")
    
    def test_rwkv_lm_repository_available(self):
        """Test that RWKV-LM repository is cloned and available"""
        rwkv_lm = self.repo_manager.get_repository("RWKV-LM")
        self.assertIsNotNone(rwkv_lm, "RWKV-LM repository should be available")
        self.assertTrue(rwkv_lm.available, "RWKV-LM repository should be marked as available")
        self.assertTrue(rwkv_lm.path.exists(), "RWKV-LM repository path should exist")
    
    def test_rwkv_lm_metadata(self):
        """Test RWKV-LM repository metadata"""
        rwkv_lm = self.repo_manager.get_repository("RWKV-LM")
        self.assertIsNotNone(rwkv_lm, "RWKV-LM repository should be available")
        
        # Check basic metadata
        self.assertEqual(rwkv_lm.repo_type, RWKVRepoType.MAIN_LM)
        self.assertIn("RWKV language model", rwkv_lm.description)
        
        # Check version metadata
        self.assertIn("versions", rwkv_lm.metadata)
        self.assertIn("v7", rwkv_lm.metadata["versions"])
        self.assertEqual(rwkv_lm.metadata["latest_version"], "v7")
        self.assertEqual(rwkv_lm.metadata["architecture"], "linear-time RNN")
        
        # Check features
        self.assertIn("features", rwkv_lm.metadata)
        features = rwkv_lm.metadata["features"]
        self.assertIn("attention-free", features)
        self.assertIn("constant-space", features)
        self.assertIn("parallelizable", features)
    
    def test_rwkv_lm_entry_points(self):
        """Test RWKV-LM entry points"""
        rwkv_lm = self.repo_manager.get_repository("RWKV-LM")
        self.assertIsNotNone(rwkv_lm, "RWKV-LM repository should be available")
        
        # Entry points should be defined (may be empty dict initially)
        self.assertIsInstance(rwkv_lm.entry_points, dict, "Entry points should be a dictionary")
        
        # Check that demo files actually exist in the repository
        v7_path = rwkv_lm.path / "RWKV-v7"
        if v7_path.exists():
            demo_files = list(v7_path.glob("*demo*.py"))
            self.assertGreater(len(demo_files), 0, "Should have demo files in RWKV-v7 directory")
    
    def test_rwkv_lm_files_exist(self):
        """Test that key RWKV-LM files exist"""
        rwkv_lm = self.repo_manager.get_repository("RWKV-LM")
        self.assertIsNotNone(rwkv_lm, "RWKV-LM repository should be available")
        
        # Check main directory exists
        self.assertTrue(rwkv_lm.path.exists(), "RWKV-LM path should exist")
        
        # Check RWKV-v7 directory exists
        v7_path = rwkv_lm.path / "RWKV-v7"
        self.assertTrue(v7_path.exists(), "RWKV-v7 directory should exist")
        
        # Check key demo files exist
        demo_file = v7_path / "rwkv_v7_demo.py"
        self.assertTrue(demo_file.exists(), "RWKV v7 demo file should exist")
        
        demo_rnn_file = v7_path / "rwkv_v7_demo_rnn.py"
        self.assertTrue(demo_rnn_file.exists(), "RWKV v7 RNN demo file should exist")
        
        # Check README exists
        readme_file = rwkv_lm.path / "README.md"
        self.assertTrue(readme_file.exists(), "RWKV-LM README should exist")
    
    def test_simple_rwkv_integration_available(self):
        """Test that SimpleRWKVIntegration works with real RWKV"""
        integration = SimpleRWKVIntegration()
        
        # Should indicate RWKV package is available even without a model
        info = integration.get_info()
        self.assertIn("rwkv_package_available", info)
        self.assertTrue(info["rwkv_package_available"], "RWKV package should be available")
        
        # Integration itself may not be available without a model, but package should be
        self.assertFalse(info["model_loaded"], "No model should be loaded initially")
        
        # Version should be retrievable
        version = integration.get_version()
        self.assertIsNotNone(version, "RWKV version should be available")
        self.assertIn("RWKV package installed", version, "Version should indicate package is installed")
    
    def test_repository_manager_initialization(self):
        """Test repository manager initialization"""
        # Should have 5 repositories configured
        self.assertEqual(len(self.repo_manager.repositories), 5)
        
        # Check that RWKV-LM is the only one available
        available_repos = [
            name for name, repo in self.repo_manager.repositories.items() 
            if repo.available
        ]
        self.assertIn("RWKV-LM", available_repos)
        
        # Other repos should not be available yet
        unavailable_repos = [
            name for name, repo in self.repo_manager.repositories.items() 
            if not repo.available
        ]
        expected_unavailable = ["ChatRWKV", "RWKV-CUDA", "WorldModel", "RWKV-v2-RNN-Pile"]
        for repo_name in expected_unavailable:
            self.assertIn(repo_name, unavailable_repos)
    
    def test_rwkv_lm_content_inspection(self):
        """Test inspection of RWKV-LM repository content"""
        rwkv_lm = self.repo_manager.get_repository("RWKV-LM")
        self.assertIsNotNone(rwkv_lm, "RWKV-LM repository should be available")
        
        # Check various version directories exist
        expected_versions = ["RWKV-v5", "RWKV-v6", "RWKV-v7"]
        for version in expected_versions:
            version_path = rwkv_lm.path / version
            self.assertTrue(version_path.exists(), f"{version} directory should exist")
        
        # Check that README contains expected content
        readme_path = rwkv_lm.path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                self.assertIn("RWKV", content)
                self.assertIn("BlinkDL", content)

def run_integration_tests():
    """Run the RWKV-LM integration tests"""
    print("🧪 Running RWKV-LM Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRWKVLMIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All RWKV-LM integration tests passed!")
        print("🎉 RWKV-LM is successfully integrated with Deep Tree Echo")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)