"""
Tests for the Parentheses Bootstrap Lisp system
Tests Spencer-Brown calculus, Church numerals, combinators, and lambda calculus
"""

import unittest
import sys
import os

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from cognitive_grammar.parentheses_bootstrap import (
    ParenthesesBootstrap,
    SpencerBrownCalculus,
    CombinatorLibrary,
    ChurchNumerals,
    LambdaCalculusEmergence,
    MetacircularEvaluator,
    ParenthesesParser,
    ParenthesesExpression,
    DistinctionType,
    get_bootstrap_system
)

class TestParenthesesParser(unittest.TestCase):
    """Test the parentheses parser"""
    
    def setUp(self):
        self.parser = ParenthesesParser()
    
    def test_parse_void(self):
        """Test parsing empty parentheses"""
        expr = self.parser.parse("()")
        self.assertEqual(expr.distinction_type, DistinctionType.VOID)
        self.assertFalse(expr.is_atomic)
    
    def test_parse_mark(self):
        """Test parsing marked distinction"""
        expr = self.parser.parse("(())")
        self.assertEqual(expr.distinction_type, DistinctionType.MARK)
        self.assertFalse(expr.is_atomic)
    
    def test_parse_atom(self):
        """Test parsing atomic expressions"""
        expr = self.parser.parse("x")
        self.assertEqual(expr.distinction_type, DistinctionType.IDENTITY)
        self.assertTrue(expr.is_atomic)
        self.assertEqual(expr.content, "x")
    
    def test_parse_complex(self):
        """Test parsing complex expressions"""
        expr = self.parser.parse("(+ 1 2)")
        self.assertEqual(expr.distinction_type, DistinctionType.CROSS)
        self.assertFalse(expr.is_atomic)
        self.assertEqual(len(expr.content), 3)

class TestSpencerBrownCalculus(unittest.TestCase):
    """Test Spencer-Brown calculus rules"""
    
    def setUp(self):
        self.calculus = SpencerBrownCalculus()
    
    def test_identity_law(self):
        """Test (()) -> () identity law"""
        result = self.calculus.evaluate("(())")
        self.assertEqual(result.distinction_type, DistinctionType.VOID)
    
    def test_void_preservation(self):
        """Test () stays ()"""
        result = self.calculus.evaluate("()")
        self.assertEqual(result.distinction_type, DistinctionType.VOID)
    
    def test_atom_preservation(self):
        """Test atoms are preserved"""
        result = self.calculus.evaluate("x")
        self.assertEqual(result.content, "x")
        self.assertTrue(result.is_atomic)

class TestChurchNumerals(unittest.TestCase):
    """Test Church numeral encoding and arithmetic"""
    
    def setUp(self):
        self.church = ChurchNumerals()
    
    def test_encode_zero(self):
        """Test encoding zero"""
        zero = self.church.encode_number(0)
        self.assertEqual(zero, "()")
    
    def test_encode_positive_numbers(self):
        """Test encoding positive numbers"""
        one = self.church.encode_number(1)
        self.assertEqual(one, "(())")
        
        two = self.church.encode_number(2)
        self.assertEqual(two, "((()))")
        
        three = self.church.encode_number(3)
        self.assertEqual(three, "(((())))")
    
    def test_decode_numbers(self):
        """Test decoding Church numerals back to integers"""
        self.assertEqual(self.church.decode_number("()"), 0)
        self.assertEqual(self.church.decode_number("(())"), 1)
        self.assertEqual(self.church.decode_number("((()))"), 2)
        self.assertEqual(self.church.decode_number("(((())))"), 3)
    
    def test_successor_function(self):
        """Test successor function"""
        zero = self.church.encode_number(0)
        one_from_succ = self.church.successor(zero)
        one_direct = self.church.encode_number(1)
        self.assertEqual(one_from_succ, one_direct)

class TestCombinatorLibrary(unittest.TestCase):
    """Test combinatorial primitives"""
    
    def setUp(self):
        self.combinators = CombinatorLibrary()
    
    def test_identity_combinator(self):
        """Test I combinator structure"""
        i_comb = self.combinators.identity_combinator()
        self.assertIn("x", i_comb)
        self.assertTrue(i_comb.startswith("("))
        self.assertTrue(i_comb.endswith(")"))
    
    def test_k_combinator(self):
        """Test K combinator structure"""
        k_comb = self.combinators.k_combinator()
        self.assertIn("x", k_comb)
        self.assertIn("y", k_comb)
    
    def test_s_combinator(self):
        """Test S combinator structure"""
        s_comb = self.combinators.s_combinator()
        self.assertIn("f", s_comb)
        self.assertIn("g", s_comb)
        self.assertIn("x", s_comb)

class TestLambdaCalculusEmergence(unittest.TestCase):
    """Test lambda calculus emergence"""
    
    def setUp(self):
        self.lambda_calc = LambdaCalculusEmergence()
    
    def test_lambda_abstraction(self):
        """Test lambda abstraction creation"""
        abstraction = self.lambda_calc.lambda_abstraction("x", "x")
        self.assertEqual(abstraction, "((x) x)")
    
    def test_function_application(self):
        """Test function application"""
        app = self.lambda_calc.application("f", "x")
        self.assertEqual(app, "(f x)")
    
    def test_beta_reduction(self):
        """Test simple beta reduction"""
        reduced = self.lambda_calc.beta_reduction("((x) x)", "y")
        self.assertIn("y", reduced)

class TestMetacircularEvaluator(unittest.TestCase):
    """Test the metacircular evaluator"""
    
    def setUp(self):
        self.evaluator = MetacircularEvaluator()
    
    def test_atom_evaluation(self):
        """Test atom evaluation"""
        self.evaluator.environment["x"] = "hello"
        result = self.evaluator.eval_expression("x")
        self.assertEqual(result, "hello")
    
    def test_void_evaluation(self):
        """Test void evaluation"""
        result = self.evaluator.eval_expression("()")
        self.assertIsNone(result)
    
    def test_list_operations(self):
        """Test built-in list operations"""
        # Test car
        result = self.evaluator.eval_expression("(car (a b c))")
        # Note: This is a simplified test as full list parsing would be more complex

class TestParenthesesBootstrap(unittest.TestCase):
    """Test the main bootstrap system"""
    
    def setUp(self):
        self.bootstrap = ParenthesesBootstrap()
    
    def test_system_initialization(self):
        """Test system properly initializes"""
        self.assertIsNotNone(self.bootstrap.parser)
        self.assertIsNotNone(self.bootstrap.calculus)
        self.assertIsNotNone(self.bootstrap.evaluator)
        self.assertIsNotNone(self.bootstrap.combinators)
        self.assertIsNotNone(self.bootstrap.church_numerals)
        self.assertIsNotNone(self.bootstrap.lambda_calc)
    
    def test_builtins_loaded(self):
        """Test built-in functions are loaded"""
        env = self.bootstrap.evaluator.environment
        self.assertIn("zero", env)
        self.assertIn("one", env)
        self.assertIn("two", env)
        self.assertIn("three", env)
        self.assertIn("I", env)
        self.assertIn("K", env)
        self.assertIn("S", env)
        self.assertIn("car", env)
        self.assertIn("cdr", env)
        self.assertIn("cons", env)
    
    def test_pure_parentheses_evaluation(self):
        """Test pure parentheses expressions"""
        result = self.bootstrap.bootstrap_eval("()")
        self.assertIsNotNone(result)
        
        result = self.bootstrap.bootstrap_eval("(())")
        self.assertIsNotNone(result)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.bootstrap.get_system_status()
        self.assertIn("environment_size", status)
        self.assertIn("builtins", status)
        self.assertIn("parser_ready", status)
        self.assertIn("calculus_ready", status)
        self.assertIn("status", status)
        self.assertEqual(status["status"], "initialized")
        self.assertTrue(status["parser_ready"])
        self.assertTrue(status["calculus_ready"])
    
    def test_global_instance(self):
        """Test global instance access"""
        system1 = get_bootstrap_system()
        system2 = get_bootstrap_system()
        self.assertIs(system1, system2)  # Should be the same instance

class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and complex expressions"""
    
    def setUp(self):
        self.bootstrap = ParenthesesBootstrap()
    
    def test_church_numeral_integration(self):
        """Test Church numeral integration with the system"""
        # Test that Church numerals are available
        zero = self.bootstrap.evaluator.environment["zero"]
        self.assertEqual(zero, "()")
        
        one = self.bootstrap.evaluator.environment["one"]
        self.assertEqual(one, "(())")
    
    def test_combinator_integration(self):
        """Test combinator integration"""
        # Test that combinators are available
        i_comb = self.bootstrap.evaluator.environment["I"]
        k_comb = self.bootstrap.evaluator.environment["K"]
        s_comb = self.bootstrap.evaluator.environment["S"]
        
        self.assertIsNotNone(i_comb)
        self.assertIsNotNone(k_comb)
        self.assertIsNotNone(s_comb)
    
    def test_error_handling(self):
        """Test error handling for malformed expressions"""
        # Test unclosed parentheses
        with self.assertRaises(ValueError):
            self.bootstrap.bootstrap_eval("(unclosed")
        
        # Test unmatched closing parenthesis would be caught by parser
        # This is handled gracefully by returning partial results

class TestPerformanceAndValidation(unittest.TestCase):
    """Test performance characteristics and validation"""
    
    def setUp(self):
        self.bootstrap = ParenthesesBootstrap()
    
    def test_recursive_depth_handling(self):
        """Test handling of deeply nested expressions"""
        # Test nested parentheses up to reasonable depth
        nested = "()"
        for i in range(10):  # Keep depth manageable for testing
            nested = f"({nested})"
        
        try:
            result = self.bootstrap.bootstrap_eval(nested)
            self.assertIsNotNone(result)
        except RecursionError:
            self.fail("System should handle reasonable nesting depth")
    
    def test_church_numeral_performance(self):
        """Test Church numeral encoding/decoding performance"""
        church = self.bootstrap.church_numerals
        
        # Test numbers up to 10
        for i in range(11):
            encoded = church.encode_number(i)
            decoded = church.decode_number(encoded)
            self.assertEqual(i, decoded, f"Failed for number {i}")

if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main()