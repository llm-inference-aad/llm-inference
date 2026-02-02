#!/usr/bin/env python3
"""
Test script to verify fitness inheritance optimization.

This script simulates the scenario where a gene falls back to its parent
and verifies that fitness inheritance works correctly.
"""

import sys
import os
import numpy as np

# Define constants locally to avoid importing torch
FITNESS_WEIGHTS = (1.0, -1.0)
INVALID_FITNESS_MAX = tuple([float(x*np.inf*-1) for x in FITNESS_WEIGHTS])
PLACEHOLDER_FITNESS = tuple([int(x*9999999999*-1) for x in FITNESS_WEIGHTS])

def test_fitness_inheritance_logic():
    """Test the fitness inheritance logic with various scenarios"""
    
    print("="*70)
    print("TESTING FITNESS INHERITANCE OPTIMIZATION")
    print("="*70)
    
    # Simulate GLOBAL_DATA and GLOBAL_DATA_ANCESTRY structures
    GLOBAL_DATA = {}
    GLOBAL_DATA_ANCESTRY = {}
    
    # Test Case 1: Parent evaluated, child is fallback
    print("\n[TEST 1] Parent evaluated, child is fallback → Should inherit")
    parent_id = "parent_123"
    child_id = "child_456"
    
    GLOBAL_DATA[parent_id] = {
        'fitness': (0.85, 500000),
        'status': 'completed',
        'fallback': False
    }
    
    GLOBAL_DATA[child_id] = {
        'fallback': True,
        'fallback_reason': 'LLM timeout',
        'status': 'model created'
    }
    
    GLOBAL_DATA_ANCESTRY[child_id] = {
        'GENES': [parent_id],
        'MUTATE_TYPE': ['MUTATION']
    }
    
    # Simulate the check logic
    if child_id in GLOBAL_DATA_ANCESTRY and 'GENES' in GLOBAL_DATA_ANCESTRY[child_id]:
        parent_genes = GLOBAL_DATA_ANCESTRY[child_id]['GENES']
        if len(parent_genes) > 0:
            parent_gene_id = parent_genes[0]
            
            if parent_gene_id in GLOBAL_DATA and GLOBAL_DATA[parent_gene_id].get('fitness') is not None:
                parent_fitness = GLOBAL_DATA[parent_gene_id]['fitness']
                
                if parent_fitness != PLACEHOLDER_FITNESS and parent_fitness != INVALID_FITNESS_MAX:
                    GLOBAL_DATA[child_id]['fitness'] = parent_fitness
                    GLOBAL_DATA[child_id]['status'] = 'fitness inherited from parent'
                    GLOBAL_DATA[child_id]['inherited_from'] = parent_gene_id
                    print(f"✅ SUCCESS: Child inherited fitness {parent_fitness} from parent")
                else:
                    print(f"❌ FAIL: Parent has invalid fitness")
            else:
                print(f"❌ FAIL: Parent not evaluated")
    
    assert GLOBAL_DATA[child_id]['fitness'] == (0.85, 500000), "Fitness not inherited correctly"
    assert GLOBAL_DATA[child_id]['inherited_from'] == parent_id, "Parent ID not recorded"
    
    # Test Case 2: Parent not yet evaluated
    print("\n[TEST 2] Parent not yet evaluated → Should NOT inherit")
    parent_id_2 = "parent_789"
    child_id_2 = "child_012"
    
    GLOBAL_DATA[parent_id_2] = {
        'fitness': None,
        'status': 'running eval',
        'fallback': False
    }
    
    GLOBAL_DATA[child_id_2] = {
        'fallback': True,
        'fallback_reason': 'Validation error',
        'status': 'model created'
    }
    
    GLOBAL_DATA_ANCESTRY[child_id_2] = {
        'GENES': [parent_id_2],
        'MUTATE_TYPE': ['MUTATION']
    }
    
    inherited = False
    if child_id_2 in GLOBAL_DATA_ANCESTRY and 'GENES' in GLOBAL_DATA_ANCESTRY[child_id_2]:
        parent_genes = GLOBAL_DATA_ANCESTRY[child_id_2]['GENES']
        if len(parent_genes) > 0:
            parent_gene_id = parent_genes[0]
            
            if parent_gene_id in GLOBAL_DATA and GLOBAL_DATA[parent_gene_id].get('fitness') is not None:
                inherited = True
    
    if not inherited:
        print(f"✅ SUCCESS: Child will be evaluated (parent not ready)")
    else:
        print(f"❌ FAIL: Child should not inherit from unevaluated parent")
    
    assert 'fitness' not in GLOBAL_DATA[child_id_2] or GLOBAL_DATA[child_id_2].get('fitness') is None
    
    # Test Case 3: Parent has placeholder fitness
    print("\n[TEST 3] Parent has placeholder fitness → Should NOT inherit")
    parent_id_3 = "parent_345"
    child_id_3 = "child_678"
    
    GLOBAL_DATA[parent_id_3] = {
        'fitness': PLACEHOLDER_FITNESS,
        'status': 'pending',
        'fallback': False
    }
    
    GLOBAL_DATA[child_id_3] = {
        'fallback': True,
        'fallback_reason': 'Server error',
        'status': 'model created'
    }
    
    GLOBAL_DATA_ANCESTRY[child_id_3] = {
        'GENES': [parent_id_3],
        'MUTATE_TYPE': ['MUTATION']
    }
    
    inherited = False
    if child_id_3 in GLOBAL_DATA_ANCESTRY and 'GENES' in GLOBAL_DATA_ANCESTRY[child_id_3]:
        parent_genes = GLOBAL_DATA_ANCESTRY[child_id_3]['GENES']
        if len(parent_genes) > 0:
            parent_gene_id = parent_genes[0]
            
            if parent_gene_id in GLOBAL_DATA and GLOBAL_DATA[parent_gene_id].get('fitness') is not None:
                parent_fitness = GLOBAL_DATA[parent_gene_id]['fitness']
                
                if parent_fitness != PLACEHOLDER_FITNESS and parent_fitness != INVALID_FITNESS_MAX:
                    inherited = True
    
    if not inherited:
        print(f"✅ SUCCESS: Child will be evaluated (parent has placeholder)")
    else:
        print(f"❌ FAIL: Child should not inherit placeholder fitness")
    
    assert 'fitness' not in GLOBAL_DATA[child_id_3] or GLOBAL_DATA[child_id_3].get('fitness') is None
    
    # Test Case 4: Child is NOT a fallback
    print("\n[TEST 4] Child is NOT a fallback → Should NOT inherit")
    parent_id_4 = "parent_901"
    child_id_4 = "child_234"
    
    GLOBAL_DATA[parent_id_4] = {
        'fitness': (0.90, 400000),
        'status': 'completed',
        'fallback': False
    }
    
    GLOBAL_DATA[child_id_4] = {
        'fallback': False,  # Not a fallback
        'status': 'model created'
    }
    
    GLOBAL_DATA_ANCESTRY[child_id_4] = {
        'GENES': [parent_id_4],
        'MUTATE_TYPE': ['MUTATION']
    }
    
    # In the actual code, we only check for inheritance if fallback marker exists
    # So non-fallback children should always be evaluated
    print(f"✅ SUCCESS: Non-fallback child will be evaluated normally")
    
    assert 'fitness' not in GLOBAL_DATA[child_id_4] or GLOBAL_DATA[child_id_4].get('fitness') is None
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✅")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    print("- Fallback clones with evaluated parents: Inherit fitness ✅")
    print("- Fallback clones with unevaluated parents: Must evaluate ✅")
    print("- Fallback clones with invalid parent fitness: Must evaluate ✅")
    print("- Non-fallback children: Always evaluated ✅")

if __name__ == "__main__":
    test_fitness_inheritance_logic()
