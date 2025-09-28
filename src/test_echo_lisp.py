"""
Test Suite for EchoLisp - Echo Structure Generation System

Validates the functionality of the EchoLisp class including:
- Successor generation logic
- Echo enumeration and tree ID tracking  
- Lisp-style string conversion
- Simulation with step-by-step evolution
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from echo_lisp import EchoLisp
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure echo_lisp.py is in the same directory")
    sys.exit(1)


class TestEchoLisp(unittest.TestCase):
    """Test cases for EchoLisp functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.echo_lisp = EchoLisp()
    
    def test_initialization(self):
        """Test EchoLisp initialization"""
        self.assertIsInstance(self.echo_lisp.treeid, dict)
        self.assertEqual(self.echo_lisp.treeid, {(): 0})
        self.assertEqual(self.echo_lisp.get_tree_id_count(), 1)
    
    def test_succ_empty_structure(self):
        """Test successor generation for empty structure"""
        successors = list(self.echo_lisp.succ(()))
        expected = [((),)]  # Only append single-node echo
        self.assertEqual(successors, expected)
    
    def test_succ_single_element(self):
        """Test successor generation for single-element structure"""
        # For (()), successors should include appended empty and recursion
        successors = list(self.echo_lisp.succ(((),)))
        
        # Should include: append empty -> ((), ()) and recursion on () -> (((),))
        self.assertIn(((), ()), successors)     # Append operation: ((),) + ((),) = ((), ())
        self.assertIn((((),),), successors)     # Recursion operation: succ(()) -> ((),) wrapped as (((),),)
        
        # Verify correct number of successors
        self.assertEqual(len(successors), 2)
    
    def test_tostr_empty(self):
        """Test string conversion for empty structure"""
        result = self.echo_lisp.tostr(())
        self.assertEqual(result, "()")
    
    def test_tostr_nested(self):
        """Test string conversion for nested structures"""
        # Test (())
        result = self.echo_lisp.tostr(((),))
        self.assertEqual(result, "(())")
        
        # Test (()())
        result = self.echo_lisp.tostr(((), ()))
        self.assertEqual(result, "(()())")
        
        # Test ((()))
        result = self.echo_lisp.tostr((((),),))
        self.assertEqual(result, "((()))")
    
    def test_echoes_size_1(self):
        """Test echo generation for size 1"""
        echoes_1 = list(self.echo_lisp.echoes(1))
        expected = [()]
        self.assertEqual(echoes_1, expected)
    
    def test_echoes_size_2(self):
        """Test echo generation for size 2"""
        echoes_2 = list(self.echo_lisp.echoes(2))
        
        # Should contain single echo: (())
        self.assertEqual(len(echoes_2), 1)
        self.assertIn(((),), echoes_2)
        
        # Check tree ID tracking
        self.assertIn(((),), self.echo_lisp.treeid)
        self.assertEqual(self.echo_lisp.get_tree_id_count(), 2)  # () and (())
    
    def test_tree_id_tracking(self):
        """Test tree ID assignment and tracking"""
        # Generate some echoes to populate tree IDs
        list(self.echo_lisp.echoes(3))
        
        # Verify initial structure is tracked
        self.assertIn((), self.echo_lisp.treeid)
        self.assertEqual(self.echo_lisp.treeid[()], 0)
        
        # Verify tree IDs are unique and sequential
        tree_ids = list(self.echo_lisp.treeid.values())
        self.assertEqual(len(tree_ids), len(set(tree_ids)))  # All unique
        self.assertEqual(min(tree_ids), 0)
        self.assertEqual(max(tree_ids), len(tree_ids) - 1)   # Sequential from 0
    
    def test_simulate_n4_expected_output(self):
        """Test simulation for n=4 matches expected output from problem statement"""
        results = self.echo_lisp.simulate(4)
        
        # Verify we have 4 steps
        self.assertEqual(len(results), 4)
        
        # Extract step numbers and structures
        steps, structures = zip(*results)
        
        # Verify step numbering
        self.assertEqual(steps, (1, 2, 3, 4))
        
        # Verify expected structures based on problem statement
        expected_structures = ["(()()())", "(()(()))", "((()()))", "(((())))" ]
        self.assertEqual(list(structures), expected_structures)
    
    def test_simulate_progressive_complexity(self):
        """Test that simulation produces increasingly complex structures"""
        results = self.echo_lisp.simulate(6)
        
        # Extract structures
        structures = [structure for _, structure in results]
        
        # Verify increasing complexity (length of string representation)
        prev_length = 0
        for structure in structures:
            current_length = len(structure)
            # Allow equal length (multiple structures of same complexity level)
            self.assertGreaterEqual(current_length, prev_length)
            prev_length = current_length
    
    def test_reset_functionality(self):
        """Test reset functionality"""
        # Generate some echoes to populate tree IDs
        list(self.echo_lisp.echoes(3))
        initial_count = self.echo_lisp.get_tree_id_count()
        
        # Verify we have more than initial state
        self.assertGreater(initial_count, 1)
        
        # Reset and verify back to initial state
        self.echo_lisp.reset()
        self.assertEqual(self.echo_lisp.treeid, {(): 0})
        self.assertEqual(self.echo_lisp.get_tree_id_count(), 1)
    
    def test_get_tree_ids_copy(self):
        """Test that get_tree_ids returns a copy, not reference"""
        list(self.echo_lisp.echoes(2))
        original_ids = self.echo_lisp.get_tree_ids()
        
        # Modify the returned copy
        original_ids[((), ())] = 999
        
        # Verify original is unchanged
        self.assertNotIn(((), ()), self.echo_lisp.treeid)
        self.assertNotEqual(self.echo_lisp.treeid[((),)], 999)
    
    def test_successor_constraint_satisfaction(self):
        """Test that successor generation satisfies tree ID constraints"""
        # Generate echoes to build tree ID mapping
        list(self.echo_lisp.echoes(4))
        
        # Test successor constraint for a multi-element structure
        test_structure = (((),), ())
        if test_structure in self.echo_lisp.treeid:
            successors = list(self.echo_lisp.succ(test_structure))
            
            # All successors should satisfy the tree ID constraint
            for successor in successors:
                if len(successor) > 1:
                    head, rest = successor[0], successor[1:]
                    if rest and rest[0] in self.echo_lisp.treeid:
                        head_id = self.echo_lisp.treeid.get(head, float('inf'))
                        rest_first_id = self.echo_lisp.treeid[rest[0]]
                        self.assertLessEqual(head_id, rest_first_id)


class TestEchoLispIntegration(unittest.TestCase):
    """Integration tests for EchoLisp with expected behaviors"""
    
    def test_demo_function_execution(self):
        """Test that the demo function executes without errors"""
        # Import the demo function
        from echo_lisp import demo_echo_lisp
        
        # Should execute without raising exceptions
        try:
            # Capture stdout to avoid cluttering test output
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                demo_echo_lisp()
                
            output = f.getvalue()
            
            # Verify demo produces expected output structure
            self.assertIn("Step 1:", output)
            self.assertIn("Step 4:", output)
            self.assertIn("(()()())", output)  # Expected first structure
            self.assertIn("(((())))", output) # Expected last structure
            
        except Exception as e:
            self.fail(f"Demo function raised exception: {e}")
    
    def test_echo_structure_uniqueness(self):
        """Test that generated echo structures are unique within each size"""
        echo_lisp = EchoLisp()
        
        for n in range(1, 6):
            echoes = list(echo_lisp.echoes(n))
            unique_echoes = list(set(echoes))
            
            # All echoes should be unique
            self.assertEqual(len(echoes), len(unique_echoes),
                           f"Duplicate echoes found for size {n}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)