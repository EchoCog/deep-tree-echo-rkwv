#!/usr/bin/env python3
"""
Demonstration of Spencer-Brown's Laws of Form using the Parentheses Bootstrap System
This script provides interactive and visual examples of the calculus in action.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cognitive_grammar import get_bootstrap_system

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a step in the demonstration"""
    print(f"\n{step}. {description}")
    print("-" * 40)

def demonstrate_primordial_distinctions():
    """Demonstrate the primordial distinctions"""
    print_header("🌌 PRIMORDIAL DISTINCTIONS")
    
    bootstrap = get_bootstrap_system()
    
    examples = [
        ("()", "The Void - unmarked state, the primordial container"),
        ("(())", "The First Mark - first distinction between inside/outside"),
        ("((()))", "Nested Mark - recursive distinction"),
        ("(((())))", "Deep Nesting - multiple levels of distinction"),
    ]
    
    print("Spencer-Brown's Laws of Form begin with the most basic distinction:")
    print("The act of drawing a boundary creates 'marked' and 'unmarked' spaces.")
    
    for i, (expr, description) in enumerate(examples, 1):
        print(f"\n{i}. {expr}")
        print(f"   → {description}")
        
        # Evaluate using our system
        result = bootstrap.bootstrap_eval(expr)
        print(f"   Evaluates to: {result}")
        
        if expr == "(())":
            print("   🎯 Notice: (()) simplifies to () via the identity law!")

def demonstrate_calculus_rules():
    """Demonstrate the fundamental calculus rules"""
    print_header("⚡ SPENCER-BROWN CALCULUS RULES")
    
    bootstrap = get_bootstrap_system()
    
    print("The Laws of Form provide two fundamental rules:")
    print("1. Identity Law: (()) → () - Double marking cancels out")
    print("2. Void Preservation: () → () - Void remains void")
    
    test_cases = [
        # Identity law examples
        ("(())", "Identity Law: Double mark becomes void"),
        ("((()))", "Nested application: Outer mark processes inner void"),
        ("(((())))", "Multiple nesting: Each level processes recursively"),
        
        # Complex expressions
        ("(()())", "Multiple elements: Each processes according to rules"),
        ("((()()))", "Nested complexity: Recursive rule application"),
    ]
    
    for expr, description in test_cases:
        print(f"\n🔄 {expr}")
        print(f"   {description}")
        
        start_time = time.time() * 1000
        result = bootstrap.bootstrap_eval(expr) 
        end_time = time.time() * 1000
        
        print(f"   Result: {result}")
        print(f"   Evaluation time: {end_time - start_time:.2f}ms")

def demonstrate_church_numerals():
    """Demonstrate Church numeral arithmetic"""
    print_header("🔢 CHURCH NUMERAL ARITHMETIC")
    
    bootstrap = get_bootstrap_system()
    church = bootstrap.church_numerals
    
    print("Numbers emerge from nested distinctions:")
    print("Each level of nesting represents the next natural number.")
    
    for n in range(5):
        encoded = church.encode_number(n)
        decoded = church.decode_number(encoded)
        
        print(f"\n{n} → {encoded}")
        print(f"   Depth: {len(encoded) - 2 if len(encoded) > 2 else 0} levels")
        print(f"   Decodes back to: {decoded}")
        
        if n == 0:
            print("   🎯 Zero is the void - no distinction")
        elif n == 1:
            print("   🎯 One is the first mark - first distinction")
    
    print("\n🔢 Successor Function:")
    print("SUCC(n) = (n) - wrapping adds one level of distinction")
    
    for n in range(3):
        original = church.encode_number(n)
        successor = church.successor(original)
        expected = church.encode_number(n + 1)
        
        print(f"\nSUCC({n}): {original} → {successor}")
        print(f"Expected {n+1}: {expected}")
        print(f"Match: {'✅' if successor == expected else '❌'}")

def demonstrate_combinators():
    """Demonstrate combinatorial logic"""
    print_header("🔄 COMBINATORIAL PRIMITIVES")
    
    bootstrap = get_bootstrap_system()
    combinators = bootstrap.combinators
    
    print("Combinators provide the computational foundation:")
    print("S, K, I combinators enable universal computation.")
    
    combinator_info = [
        ("I", "Identity", "I x = x", combinators.identity_combinator()),
        ("K", "Constant", "K x y = x", combinators.k_combinator()),
        ("S", "Substitution", "S f g x = (f x (g x))", combinators.s_combinator()),
    ]
    
    for name, full_name, rule, implementation in combinator_info:
        print(f"\n🔧 {name} Combinator ({full_name})")
        print(f"   Rule: {rule}")
        print(f"   Implementation: {implementation}")
        
        # Show in environment
        env_value = bootstrap.evaluator.environment.get(name, "Not found")
        print(f"   Available as: {name} = {env_value}")

def demonstrate_lambda_emergence():
    """Demonstrate lambda calculus emergence"""
    print_header("λ LAMBDA CALCULUS EMERGENCE")
    
    bootstrap = get_bootstrap_system()
    lambda_calc = bootstrap.lambda_calc
    
    print("Lambda calculus emerges from parenthetical structure:")
    print("Functions are just special arrangements of distinctions.")
    
    examples = [
        ("identity", "x", "x", "Identity function - returns its argument"),
        ("const", "x", "y", "Constant function - ignores second argument"),
        ("apply", "f", "(f x)", "Function application"),
    ]
    
    for name, var, body, description in examples:
        abstraction = lambda_calc.lambda_abstraction(var, body)
        print(f"\n🔗 {name.title()} Function")
        print(f"   Mathematical: λ{var}.{body}")
        print(f"   Parenthetical: {abstraction}")
        print(f"   Description: {description}")
        
        # Show application example
        if name == "identity":
            application = lambda_calc.application(abstraction, "hello")
            print(f"   Application: ({abstraction} hello) = {application}")

def demonstrate_metacircular_evaluation():
    """Demonstrate the metacircular evaluator"""
    print_header("🔄 METACIRCULAR EVALUATION")
    
    bootstrap = get_bootstrap_system()
    evaluator = bootstrap.evaluator
    
    print("The system can evaluate its own expressions:")
    print("This creates a self-contained computational environment.")
    
    # Show environment contents
    print(f"\n📚 Built-in Environment ({len(evaluator.environment)} items):")
    for key, value in list(evaluator.environment.items())[:8]:
        if callable(value):
            print(f"   {key}: <function>")
        else:
            print(f"   {key}: {str(value)[:20]}{'...' if len(str(value)) > 20 else ''}")
    
    if len(evaluator.environment) > 8:
        print(f"   ... and {len(evaluator.environment) - 8} more")
    
    # Show evaluation examples
    print("\n🎯 Evaluation Examples:")
    
    simple_examples = [
        "zero",      # Built-in church numeral
        "one",       # Built-in church numeral
        "I",         # Identity combinator
        "()",        # Pure void
        "(())",      # Pure mark
    ]
    
    for expr in simple_examples:
        try:
            result = evaluator.eval_expression(expr)
            print(f"   {expr} → {result}")
        except Exception as e:
            print(f"   {expr} → Error: {e}")

def demonstrate_system_status():
    """Show comprehensive system status"""
    print_header("📊 SYSTEM STATUS & CAPABILITIES")
    
    bootstrap = get_bootstrap_system()
    status = bootstrap.get_system_status()
    
    print("🚀 Parentheses Bootstrap System Status:")
    for key, value in status.items():
        if isinstance(value, list):
            print(f"   {key}: {len(value)} items")
            if key == 'builtins' and len(value) > 0:
                print(f"      Examples: {', '.join(value[:5])}...")
        else:
            print(f"   {key}: {value}")
    
    print("\n✨ Capabilities Demonstrated:")
    capabilities = [
        "✅ Spencer-Brown Laws of Form calculus",
        "✅ Church numeral arithmetic via containment", 
        "✅ Combinatorial logic (S, K, I combinators)",
        "✅ Lambda calculus emergence from parentheses",
        "✅ Metacircular evaluation environment",
        "✅ Self-modifying code potential",
        "✅ Integration with cognitive architecture"
    ]
    
    for capability in capabilities:
        print(f"   {capability}")

def interactive_mode():
    """Run interactive mode for user experimentation"""
    print_header("🎮 INTERACTIVE MODE")
    
    bootstrap = get_bootstrap_system()
    
    print("Enter parenthetical expressions to evaluate them.")
    print("Type 'help' for examples, 'status' for system info, or 'quit' to exit.")
    
    while True:
        try:
            user_input = input("\n🧠 bootstrap> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\n📖 Example expressions to try:")
                print("   ()           - The void")
                print("   (())         - First mark (becomes void)")
                print("   ((()))       - Nested mark")
                print("   zero         - Church numeral 0")
                print("   one          - Church numeral 1") 
                print("   I            - Identity combinator")
                print("   (+ 1 2)      - Complex expression")
            elif user_input.lower() == 'status':
                status = bootstrap.get_system_status()
                print(f"\n📊 System Status: {status['status']}")
                print(f"   Environment: {status['environment_size']} bindings")
                print(f"   Parser: {'Ready' if status['parser_ready'] else 'Not Ready'}")
                print(f"   Calculus: {'Ready' if status['calculus_ready'] else 'Not Ready'}")
            elif user_input:
                start_time = time.time() * 1000
                try:
                    result = bootstrap.bootstrap_eval(user_input)
                    end_time = time.time() * 1000
                    print(f"→ {result}")
                    print(f"  (evaluated in {end_time - start_time:.2f}ms)")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

def main():
    """Main demonstration function"""
    print("🧠 Deep Tree Echo: Spencer-Brown Laws of Form Demonstration")
    print("═══════════════════════════════════════════════════════════")
    print("\nThis demonstration shows how Lisp emerges from pure parentheses")
    print("using G. Spencer-Brown's Laws of Form as implemented in our")
    print("Parentheses Bootstrap System.")
    
    demonstrations = [
        ("1", "Primordial Distinctions", demonstrate_primordial_distinctions),
        ("2", "Calculus Rules", demonstrate_calculus_rules), 
        ("3", "Church Numerals", demonstrate_church_numerals),
        ("4", "Combinators", demonstrate_combinators),
        ("5", "Lambda Emergence", demonstrate_lambda_emergence),
        ("6", "Metacircular Evaluation", demonstrate_metacircular_evaluation),
        ("7", "System Status", demonstrate_system_status),
        ("i", "Interactive Mode", interactive_mode),
    ]
    
    while True:
        print("\n🎯 Available Demonstrations:")
        for key, title, _ in demonstrations:
            print(f"   {key}. {title}")
        print("   q. Quit")
        
        choice = input("\nSelect demonstration (or 'all' for complete demo): ").strip().lower()
        
        if choice in ['q', 'quit', 'exit']:
            print("👋 Thanks for exploring Spencer-Brown's Laws of Form!")
            break
        elif choice == 'all':
            for key, title, demo_func in demonstrations[:-1]:  # Skip interactive for 'all'
                demo_func()
                time.sleep(1)  # Brief pause between demonstrations
            break
        else:
            # Find and run specific demonstration
            demo_found = False
            for key, title, demo_func in demonstrations:
                if choice == key:
                    demo_func()
                    demo_found = True
                    break
            
            if not demo_found:
                print("❌ Invalid choice. Please try again.")

if __name__ == '__main__':
    main()