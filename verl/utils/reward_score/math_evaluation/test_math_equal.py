# -*- coding: utf-8 -*-
"""
Test math_equal() function input format and comparison scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grader import math_equal

def print_test_case(test_name, prediction, reference, expected, **kwargs):
    """
    Print test case and result
    """
    try:
        result = math_equal(prediction, reference, **kwargs)
        status = "PASS" if result == expected else "FAIL"
    except Exception as e:
        result = None
        status = "ERROR"
        print(f"\n[{status}] {test_name}")
        print(f"    Prediction: {repr(prediction)}")
        print(f"    Reference:  {repr(reference)}")
        print(f"    Error: {e}")
        return False
    
    print(f"\n[{status}] {test_name}")
    print(f"    Prediction: {repr(prediction)}")
    print(f"    Reference:  {repr(reference)}")
    print(f"    Result: {result}, Expected: {expected}")
    if kwargs:
        print(f"    Options: {kwargs}")
    return result == expected


def main():
    print("="*80)
    print("Testing math_equal() Function")
    print("="*80)
    print("\nFunction Signature:")
    print("  math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=False)")
    print("\nParameter Explanation:")
    print("  - prediction: Union[bool, float, str]  # Model generated answer")
    print("  - reference: Union[float, str]          # Standard/ground truth answer")
    print("  - include_percentage: bool              # Check percentage form (default True)")
    print("  - is_close: bool                        # Use approximate equality (default True)")
    print("  - timeout: bool                         # Use timeout control for symbolic comparison (default False)")
    print("  Return: bool                            # Whether equal")
    
    passed = 0
    total = 0
    
    # ========== Category 1: Numerical Equality ==========
    print("\n" + "="*80)
    print("Category 1: Numerical Equality Test")
    print("="*80)
    
    test_cases_numeric = [
        ("Integer equal", "42", "42", True),
        ("Float equal", "3.14", "3.14", True),
        ("Float approximate", "3.1400001", "3.14", True),
        ("Scientific notation", "1e-3", "0.001", True),
        ("Number with comma", "1,000", "1000", True),
        ("Percentage form", "50%", "0.5", True),
        ("Different numbers", "3.14", "2.71", False),
        ("Int vs float", "2", "2.0", True),
        ("Negative number", "-5.5", "-5.5", True),
        ("Different negatives", "-5.5", "5.5", False),
    ]
    
    for test_name, pred, ref, expected in test_cases_numeric:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 2: String Comparison ==========
    print("\n" + "="*80)
    print("Category 2: String Comparison (case insensitive)")
    print("="*80)
    
    test_cases_string = [
        ("Simple string equal", "hello", "hello", True),
        ("Case insensitive", "HELLO", "hello", True),
        ("Whitespace handling", "  hello  ", "hello  ", True),
        ("Different strings", "hello", "world", False),
    ]
    
    for test_name, pred, ref, expected in test_cases_string:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 3: Multiple Choice Format ==========
    print("\n" + "="*80)
    print("Category 3: Multiple Choice Format (A/B/C/D/E)")
    print("="*80)
    
    test_cases_choice = [
        ("Option A", "A", "A", True),
        ("Option B", "B", "B", True),
        ("Space around option", "  C  ", "C", True),
        ("Punctuation before", ": D", "D", True),
        ("Punctuation after", "E.", "E", True),
        ("Case insensitive", "a", "A", True),
        ("Different options", "A", "B", False),
        ("Non-option text", "apple", "A", False),
    ]
    
    for test_name, pred, ref, expected in test_cases_choice:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 4: List/Tuple Format ==========
    print("\n" + "="*80)
    print("Category 4: List/Tuple Format [a,b] vs [c,d]")
    print("="*80)
    
    test_cases_list = [
        ("List equal", "[1, 2]", "[1, 2]", True),
        ("Tuple equal", "(1, 2)", "(1, 2)", True),
        ("List vs tuple", "[1, 2]", "(1, 2)", True),
        ("Numeric approximate", "(1.0, 2.0)", "(1, 2)", True),
        ("Different length", "[1, 2]", "[1, 2, 3]", False),
        ("Different elements", "[1, 2]", "[1, 3]", False),
        ("Fraction list", "[1/2, 3/4]", "[0.5, 0.75]", True),
    ]
    
    for test_name, pred, ref, expected in test_cases_list:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 5: LaTeX and Symbolic ==========
    print("\n" + "="*80)
    print("Category 5: LaTeX and Symbolic Math Expressions")
    print("="*80)
    
    test_cases_latex = [
        ("Fraction equal", r"\frac{1}{2}", "0.5", True),
        ("Same fraction", r"\frac{2}{4}", r"\frac{1}{2}", True),
        ("Simplified form", r"x^2 + 2x + 1", r"(x + 1)^2", True),
    ]
    
    for test_name, pred, ref, expected in test_cases_latex:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 6: Equation Format ==========
    print("\n" + "="*80)
    print("Category 6: Equation Format")
    print("="*80)
    
    test_cases_equation = [
        ("Equation equal", "x = 5", "x = 5", True),
        ("Extract right side", "y = 3.5", "3.5", True),
        ("Extract left side", "5", "x = 5", True),
    ]
    
    for test_name, pred, ref, expected in test_cases_equation:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 7: Edge Cases ==========
    print("\n" + "="*80)
    print("Category 7: Edge Cases and Special Values")
    print("="*80)
    
    test_cases_edge = [
        ("None input", None, "5", False),
        ("Both None", None, None, False),
        ("Empty string", "", "", True),
        ("Zero", "0", "0", True),
        ("Float zero", "0.0", "0", True),
    ]
    
    for test_name, pred, ref, expected in test_cases_edge:
        total += 1
        if print_test_case(test_name, pred, ref, expected):
            passed += 1
    
    # ========== Category 8: Parameter Options ==========
    print("\n" + "="*80)
    print("Category 8: Parameter Options Test")
    print("="*80)
    
    # Without percentage
    total += 1
    result = print_test_case(
        "No percentage check (include_percentage=False)",
        "50%", "0.5", False,
        include_percentage=False
    )
    if result:
        passed += 1
    
    # Exact comparison
    total += 1
    result = print_test_case(
        "Exact comparison (is_close=False)",
        "3.1400001", "3.14", False,
        is_close=False
    )
    if result:
        passed += 1
    
    # ========== Final Summary ==========
    print("\n" + "="*80)
    print(f"Test Summary: {passed}/{total} passed")
    print("="*80)
    
    if passed == total:
        print("SUCCESS: All tests passed!")
    else:
        print(f"FAILURE: {total - passed} tests failed")


if __name__ == "__main__":
    main()
