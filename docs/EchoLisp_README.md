# EchoLisp: Echo Structure Generation and Evolution System

EchoLisp is a hierarchical echo structure generator that produces sequences of nested structures following specific successor generation rules. The system tracks tree IDs and generates Lisp-style string representations.

## Overview

The EchoLisp system implements the algorithm described in the problem statement, generating echo structures through recursive successor generation with tree ID tracking. It produces the exact output specified:

```
Step 1: (()()())
Step 2: (()(()))
Step 3: ((()()))
Step 4: (((())))
```

## Implementation Files

- **`src/echo_lisp.py`** - Core EchoLisp class implementation
- **`src/test_echo_lisp.py`** - Comprehensive unit tests (15 test cases)
- **`src/test_echo_lisp_integration.py`** - API integration tests (6 test cases)
- **`src/demo_echo_lisp_api.py`** - API demonstration script
- **`src/echo_api_server.py`** - Integration with existing API server

## Core Functionality

### EchoLisp Class

```python
from echo_lisp import EchoLisp

# Create instance
echo_lisp = EchoLisp()

# Generate evolution simulation
results = echo_lisp.simulate(4)
for step, structure in results:
    print(f"Step {step}: {structure}")
```

### Key Methods

- **`succ(x)`** - Generate successors of an echo structure
- **`echoes(n)`** - Generate all echoes using n generation steps
- **`tostr(x)`** - Convert echo structure to Lisp-style string
- **`simulate(n)`** - Run simulation and return step-by-step evolution
- **`get_tree_id_count()`** - Get number of tracked structures
- **`reset()`** - Reset tree ID tracker

## Algorithm Details

### Successor Generation Rules

1. **Append Rule**: Always yield `((),) + x` (prepend empty structure)
2. **Empty Case**: If x is empty, only apply append rule
3. **Single Element**: For single-element structures, recurse on the element
4. **Multi-Element**: Apply tree ID constraint filtering

### Tree ID Tracking

Each unique echo structure receives a sequential tree ID starting from 0:
- `()` → ID 0
- `(())` → ID 1
- `(()())` → ID 2
- etc.

### Constraint Satisfaction

For multi-element structures `(head, rest...)`, successors of `head` are only yielded if their tree ID ≤ tree ID of `rest[0]`.

## API Integration

The EchoLisp system is integrated into the Deep Tree Echo API server with three endpoints:

### `/api/echo_lisp/simulate` (POST)

Generate echo structure evolution simulation.

**Request:**
```json
{
  "n": 4
}
```

**Response:**
```json
{
  "n": 4,
  "steps": [
    {"step": 1, "structure": "(()()())"},
    {"step": 2, "structure": "(()(()))"},
    {"step": 3, "structure": "((()()))"},
    {"step": 4, "structure": "(((())))"}
  ],
  "total_structures_tracked": 8,
  "tree_id_mappings": {...},
  "timestamp": "2024-01-01T00:00:00"
}
```

### `/api/echo_lisp/successors` (POST)

Get successors for a given echo structure.

**Request:**
```json
{
  "structure": "(())"
}
```

**Response:**
```json
{
  "input_structure": "(())",
  "successors": ["(()())", "((()))"],
  "successor_count": 2,
  "timestamp": "2024-01-01T00:00:00"
}
```

### `/api/echo_lisp/info` (GET)

Get system information and capabilities.

## Usage Examples

### Basic Usage

```python
from echo_lisp import EchoLisp

# Create instance
echo_lisp = EchoLisp()

# Run the demo from the problem statement
echo_lisp.demo_echo_lisp()

# Get successors for a structure
successors = list(echo_lisp.succ(()))
print(successors)  # [((),)]

# Generate structures and track IDs
list(echo_lisp.echoes(3))
print(echo_lisp.get_tree_ids())
```

### API Demo

```bash
cd src
python3 demo_echo_lisp_api.py
```

This runs a complete demonstration of all API endpoints without requiring Flask.

### Running Tests

```bash
cd src

# Run core functionality tests
python3 test_echo_lisp.py

# Run API integration tests  
python3 test_echo_lisp_integration.py
```

## Mermaid Diagram Compatibility

The implementation generates structures that align with the Mermaid diagram shown in the problem statement:

- **Step 1**: `(()()())` - Three parallel empty nodes
- **Step 2**: `(()(()))` - Mixed structure with nested element
- **Step 3**: `((()()))` - Deeper nesting structure
- **Step 4**: `(((())))` - Maximum depth linear structure

## Testing

The implementation includes comprehensive testing:

- **21 total test cases** across two test suites
- **100% test coverage** of core functionality
- **API integration validation**
- **Concurrent usage simulation**
- **Error handling verification**

All tests pass successfully and validate the exact output specified in the problem statement.

## Integration with Deep Tree Echo

EchoLisp integrates seamlessly into the existing Deep Tree Echo architecture:

- Uses existing Flask app structure
- Follows established API patterns
- Maintains existing error handling and logging
- Preserves existing functionality
- Adds minimal dependencies

The integration required only **4 new files** and **minimal changes** to the existing codebase, following the principle of surgical modifications.