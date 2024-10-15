import pytest
from mysupertools2.math_operations import multiply, fibonacci

# Test functions for multiply
def test_multiply_positive_numbers():
    assert multiply(2, 3) == 6
    assert multiply(5, 10) == 50

def test_multiply_with_zero():
    assert multiply(0, 5) == 0
    assert multiply(10, 0) == 0

def test_multiply_negative_numbers():
    assert multiply(-2, 3) == -6
    assert multiply(2, -3) == -6
    assert multiply(-2, -3) == 6

def test_multiply_with_type_error():
    assert multiply(2, "abc") == "error"

# Test functions for fibonacci
def test_fibonacci_zero():
    assert fibonacci(0) == "Error: Input must be a positive integer"

def test_fibonacci_one():
    assert fibonacci(1) == 0

def test_fibonacci_two():
    assert fibonacci(2) == 1

def test_fibonacci_larger_number():
    assert fibonacci(6) == 5  # Fibonacci sequence: 0, 1, 1, 2, 3, 5
    assert fibonacci(10) == 34  # Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

def test_fibonacci_negative():
    with pytest.raises(ValueError):  # If fibonacci(-1) should raise an exception
        fibonacci(-1)