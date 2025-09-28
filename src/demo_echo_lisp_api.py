"""
Demo of EchoLisp API Integration

Demonstrates the functionality that would be available through the API endpoints
without requiring Flask to be installed.
"""

import json
from datetime import datetime
from echo_lisp import EchoLisp


def simulate_api_simulate_endpoint(n=4):
    """Simulate the /api/echo_lisp/simulate endpoint"""
    print(f"\n=== Simulating POST /api/echo_lisp/simulate with n={n} ===")
    
    try:
        # Create EchoLisp instance and run simulation
        echo_lisp = EchoLisp()
        results = echo_lisp.simulate(n)
        
        # Format results as API would
        steps = []
        for step, structure in results:
            steps.append({
                'step': step,
                'structure': structure
            })
        
        response = {
            'n': n,
            'steps': steps,
            'total_structures_tracked': echo_lisp.get_tree_id_count(),
            'tree_id_mappings': {
                echo_lisp.tostr(structure): tree_id 
                for structure, tree_id in echo_lisp.get_tree_ids().items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print("Response:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def simulate_api_successors_endpoint(structure='(())'):
    """Simulate the /api/echo_lisp/successors endpoint"""
    print(f"\n=== Simulating POST /api/echo_lisp/successors with structure='{structure}' ===")
    
    try:
        # Create EchoLisp instance
        echo_lisp = EchoLisp()
        
        # Simple structure parser for demo
        structure_map = {
            '()': (),
            '(())': ((),),
            '(()())': ((), ()),
            '((()))': (((),),),
            '(()()())': ((), (), ()),
            '(()(()))': ((), ((),)),
            '((()()))': ((((),),),),
            '(((()))': ((((),),),)
        }
        
        if structure not in structure_map:
            raise ValueError(f'Unsupported structure format: {structure}')
        
        structure_tuple = structure_map[structure]
        
        # Generate enough structures to populate tree IDs
        list(echo_lisp.echoes(5))
        
        # Generate successors
        successors = list(echo_lisp.succ(structure_tuple))
        
        response = {
            'input_structure': structure,
            'successors': [echo_lisp.tostr(succ) for succ in successors],
            'successor_count': len(successors),
            'timestamp': datetime.now().isoformat()
        }
        
        print("Response:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")


def simulate_api_info_endpoint():
    """Simulate the /api/echo_lisp/info endpoint"""
    print("\n=== Simulating GET /api/echo_lisp/info ===")
    
    response = {
        'available': True,
        'description': 'Echo Structure Generation and Evolution System',
        'capabilities': [
            'Generate echo structure evolution simulations',
            'Calculate successors for echo structures',
            'Track tree IDs for generated structures',
            'Convert structures to Lisp-style string representation'
        ],
        'endpoints': {
            '/api/echo_lisp/simulate': 'POST - Generate echo evolution simulation',
            '/api/echo_lisp/successors': 'POST - Get successors for a structure',
            '/api/echo_lisp/info': 'GET - Get system information'
        },
        'example_usage': {
            'simulate': {
                'method': 'POST',
                'url': '/api/echo_lisp/simulate',
                'body': {'n': 4},
                'description': 'Generate 4-step echo evolution'
            },
            'successors': {
                'method': 'POST', 
                'url': '/api/echo_lisp/successors',
                'body': {'structure': '(())'},
                'description': 'Get successors for structure (())'
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    print("Response:")
    print(json.dumps(response, indent=2))


def main():
    """Run all API endpoint demonstrations"""
    print("🌳 EchoLisp API Integration Demo 🌳")
    print("Demonstrating the functionality available through the API endpoints")
    
    # Demonstrate info endpoint
    simulate_api_info_endpoint()
    
    # Demonstrate simulate endpoint with n=4 (matches problem statement)
    simulate_api_simulate_endpoint(4)
    
    # Demonstrate simulate endpoint with different n
    simulate_api_simulate_endpoint(6)
    
    # Demonstrate successors endpoint
    simulate_api_successors_endpoint('()')
    simulate_api_successors_endpoint('(())')
    simulate_api_successors_endpoint('(()())')
    
    print("\n✨ Demo completed successfully!")
    print("\nTo use these endpoints with the actual API server:")
    print("1. Install Flask: pip install flask flask-cors")
    print("2. Run: python echo_api_server.py")
    print("3. Make HTTP requests to http://localhost:8000/api/echo_lisp/...")


if __name__ == "__main__":
    main()