import pytest
from mysupertools2.math_operations import multiply, fibonacci

def test_multiply_0x1():
    assert multiply(0, 1) == 0

def test_multiply_1x1():
    assert multiply(1, 1) == 1

def test_multiply_2x3():
    assert multiply(2, 3) == 6

def test_multiply_ax1():
    assert multiply('a', 1) == 'error'

def test_fibonacci0():
    assert fibonacci(0) == 0

def test_fibonacci1():
    assert fibonacci(1) == 1

def test_fibonacci10():
    assert fibonacci(10) == 34

def test_fibonacci_neg():
    assert fibonacci(-1) == 'error'