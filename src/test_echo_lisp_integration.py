"""
Test Suite for EchoLisp API Integration

Tests the integration of EchoLisp with the Deep Tree Echo API server
without requiring Flask to be installed.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from echo_lisp import EchoLisp
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestEchoLispAPIIntegration(unittest.TestCase):
    """Test EchoLisp integration functionality without Flask dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.echo_lisp = EchoLisp()
    
    def test_simulate_api_functionality(self):
        """Test the core functionality that would be used by the API simulate endpoint"""
        # Test the functionality that the API endpoint would use
        n = 4
        results = self.echo_lisp.simulate(n)
        
        # Format as the API would
        steps = []
        for step, structure in results:
            steps.append({
                'step': step,
                'structure': structure
            })
        
        # Verify API-style response structure
        response_data = {
            'n': n,
            'steps': steps,
            'total_structures_tracked': self.echo_lisp.get_tree_id_count(),
            'tree_id_mappings': {
                self.echo_lisp.tostr(structure): tree_id 
                for structure, tree_id in self.echo_lisp.get_tree_ids().items()
            }
        }
        
        # Validate response structure
        self.assertEqual(response_data['n'], 4)
        self.assertEqual(len(response_data['steps']), 4)
        self.assertIsInstance(response_data['total_structures_tracked'], int)
        self.assertIsInstance(response_data['tree_id_mappings'], dict)
        
        # Validate step format
        for i, step_data in enumerate(response_data['steps']):
            self.assertEqual(step_data['step'], i + 1)
            self.assertIsInstance(step_data['structure'], str)
            self.assertTrue(step_data['structure'].startswith('('))
            self.assertTrue(step_data['structure'].endswith(')'))
        
        # Validate expected structures
        expected_structures = ["(()()())", "(()(()))", "((()()))", "(((())))" ]
        actual_structures = [step['structure'] for step in response_data['steps']]
        self.assertEqual(actual_structures, expected_structures)
    
    def test_successors_api_functionality(self):
        """Test the core functionality that would be used by the API successors endpoint"""
        # Test structure mapping that the API would use
        structure_map = {
            '()': (),
            '(())': ((),),
            '(()())': ((), ()),
            '((()))': (((),),),
        }
        
        # Need to populate tree IDs first by generating echoes
        list(self.echo_lisp.echoes(5))  # Generate enough structures to populate tree IDs
        
        # Test successors for each structure
        for structure_str, structure in structure_map.items():
            if structure in self.echo_lisp.treeid:  # Only test if structure exists in tree IDs
                successors = list(self.echo_lisp.succ(structure))
                successor_strings = [self.echo_lisp.tostr(succ) for succ in successors]
                
                # Verify API response format
                response_data = {
                    'input_structure': structure_str,
                    'successors': successor_strings,
                    'successor_count': len(successors)
                }
                
                self.assertEqual(response_data['input_structure'], structure_str)
                self.assertEqual(response_data['successor_count'], len(successor_strings))
                self.assertEqual(len(response_data['successors']), len(successors))
                
                # Verify all successors are valid Lisp strings
                for succ_str in response_data['successors']:
                    self.assertTrue(succ_str.startswith('('))
                    self.assertTrue(succ_str.endswith(')'))
    
    def test_tree_id_mapping_serialization(self):
        """Test that tree ID mappings can be properly serialized for API responses"""
        # Generate some structures
        list(self.echo_lisp.echoes(3))
        
        # Create mapping as API would
        tree_id_mappings = {
            self.echo_lisp.tostr(structure): tree_id 
            for structure, tree_id in self.echo_lisp.get_tree_ids().items()
        }
        
        # Verify all keys are strings and values are integers
        for structure_str, tree_id in tree_id_mappings.items():
            self.assertIsInstance(structure_str, str)
            self.assertIsInstance(tree_id, int)
            self.assertTrue(structure_str.startswith('('))
            self.assertTrue(structure_str.endswith(')'))
            self.assertGreaterEqual(tree_id, 0)
        
        # Verify empty structure is always present with ID 0
        self.assertIn('()', tree_id_mappings)
        self.assertEqual(tree_id_mappings['()'], 0)
    
    def test_api_input_validation_logic(self):
        """Test input validation logic that would be used by API endpoints"""
        # Test valid n values - note that simulate(n) doesn't generate exactly n results
        # but rather the first n structures from the echo generation sequence
        test_cases = [
            (1, 1),  # n=1 generates 1 result
            (2, 1),  # n=2 generates 1 result  
            (3, 2),  # n=3 generates 2 results
            (4, 4),  # n=4 generates 4 results
            (5, 9),  # n=5 generates 9 results
        ]
        
        for n, expected_count in test_cases:
            results = self.echo_lisp.simulate(n)
            self.assertEqual(len(results), expected_count, f"For n={n}, expected {expected_count} results")
            # Reset for next test
            self.echo_lisp.reset()
        
        # Test edge cases for structure mapping
        valid_structures = ['()', '(())', '(()())', '((()))']
        structure_map = {
            '()': (),
            '(())': ((),),
            '(()())': ((), ()),
            '((()))': (((),),),
        }
        
        for structure_str in valid_structures:
            self.assertIn(structure_str, structure_map)
            structure = structure_map[structure_str]
            # Make sure the structure is in tree IDs first
            echo_lisp = EchoLisp()
            list(echo_lisp.echoes(4))  # Generate enough structures
            if structure in echo_lisp.treeid:
                successors = list(echo_lisp.succ(structure))
                self.assertIsInstance(successors, list)
                self.assertGreater(len(successors), 0)
    
    def test_api_error_handling_scenarios(self):
        """Test scenarios that would cause API errors"""
        # Test what happens with moderately large n
        # n=8 generates 115 structures, which is fine
        try:
            results = self.echo_lisp.simulate(8)
            # Just verify it doesn't crash, the count will vary
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
        except Exception as e:
            self.fail(f"EchoLisp should handle n=8 without crashing: {e}")
        
        # Test empty successor generation doesn't crash
        try:
            successors = list(self.echo_lisp.succ(()))
            self.assertGreater(len(successors), 0)
        except Exception as e:
            self.fail(f"Successor generation should not crash: {e}")
    
    def test_concurrent_usage_simulation(self):
        """Test that multiple EchoLisp instances can work independently (simulating concurrent API requests)"""
        # Create multiple instances as would happen with concurrent API requests
        echo_lisp1 = EchoLisp()
        echo_lisp2 = EchoLisp()
        
        # Run different simulations - use expected counts
        results1 = echo_lisp1.simulate(3)  # Generates 2 results
        results2 = echo_lisp2.simulate(5)  # Generates 9 results
        
        # Verify they don't interfere with each other
        self.assertEqual(len(results1), 2)
        self.assertEqual(len(results2), 9)
        
        # Verify their tree ID counts are independent
        count1 = echo_lisp1.get_tree_id_count()
        count2 = echo_lisp2.get_tree_id_count()
        
        # count2 should be larger since it generated more echoes
        self.assertGreater(count2, count1)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)