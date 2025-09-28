"""
EchoLisp: Echo Structure Generation and Evolution System

Implements a hierarchical echo structure generator that produces sequences
of nested structures following specific successor generation rules.
The system tracks tree IDs and generates Lisp-style string representations.
"""


class EchoLisp:
    """
    Echo Structure Generator implementing recursive successor generation
    with tree ID tracking and Lisp-style output formatting.
    """
    
    def __init__(self):
        """Initialize the EchoLisp system with empty tree ID tracker."""
        self.treeid = {(): 0}  # Echo ID tracker - maps structures to unique IDs
    
    def succ(self, x):
        """
        Generate successors of an echo structure.
        
        Args:
            x: A tuple representing an echo structure
            
        Yields:
            Tuple: Successor echo structures
        """
        # Always yield the structure with an appended single-node echo
        yield ((),) + x
        
        if not x:
            return  # If x is empty, only the append operation applies
        
        if len(x) == 1:
            # For single-element structures, recurse on the element
            for i in self.succ(x[0]):
                yield (i,)
            return
        
        # For multi-element structures, process head and rest
        head, rest = x[0], tuple(x[1:])
        top = self.treeid.get(rest[0], float('inf'))
        
        # Generate successors where tree ID constraint is satisfied
        for i in [i for i in self.succ(head) if self.treeid.get(i, float('inf')) <= top]:
            yield (i,) + rest
    
    def echoes(self, n):
        """
        Generate all echoes of size n.
        
        Args:
            n: Size of echoes to generate
            
        Yields:
            Tuple: Echo structures of size n
        """
        if n == 1:
            yield ()
            return
        
        # Generate echoes recursively
        for x in self.echoes(n - 1):
            for a in self.succ(x):
                # Track new structures with unique tree IDs
                if a not in self.treeid:
                    self.treeid[a] = len(self.treeid)
                yield a
    
    def tostr(self, x):
        """
        Convert echo structure to a readable Lisp-style string.
        
        Args:
            x: A tuple representing an echo structure
            
        Returns:
            str: Lisp-style string representation
        """
        if not isinstance(x, tuple):
            return str(x)
        return "(" + "".join(map(self.tostr, x)) + ")"
    
    def simulate(self, n):
        """
        Simulate and display echo evolution.
        
        Args:
            n: Number of simulation steps
            
        Returns:
            List[Tuple[int, str]]: List of (step_number, structure_string) pairs
        """
        results = []
        for step, x in enumerate(self.echoes(n)):
            results.append((step + 1, self.tostr(x)))
        return results
    
    def get_tree_id_count(self):
        """
        Get the current number of tracked tree IDs.
        
        Returns:
            int: Number of unique structures tracked
        """
        return len(self.treeid)
    
    def get_tree_ids(self):
        """
        Get a copy of the current tree ID mapping.
        
        Returns:
            dict: Copy of the tree ID mapping
        """
        return self.treeid.copy()
    
    def reset(self):
        """Reset the tree ID tracker to initial state."""
        self.treeid = {(): 0}


def demo_echo_lisp():
    """Demonstrate EchoLisp functionality with n=4 simulation."""
    print("=== EchoLisp Demo: Echo Structure Evolution ===\n")
    
    # Instantiate and run the simulation
    echolisp = EchoLisp()
    steps = echolisp.simulate(4)
    
    # Print step-by-step evolution
    print("Step-by-step evolution:")
    for step, structure in steps:
        print(f"Step {step}: {structure}")
    
    print(f"\nTotal unique structures tracked: {echolisp.get_tree_id_count()}")
    print("\nTree ID mappings:")
    for structure, tree_id in echolisp.get_tree_ids().items():
        structure_str = echolisp.tostr(structure)
        print(f"  {structure_str} -> ID {tree_id}")


if __name__ == "__main__":
    demo_echo_lisp()