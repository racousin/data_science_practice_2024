import pytest
from mysupertools2.math_operations import multiply, fibonacci

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 2) == 0
    assert multiply('a', 5) == 'error'
    assert multiply(2.0, 1.2) == 2.4



def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(8) == 21


def test_fibonacci_negative():
    with pytest.raises(ValueError):
        fibonacci(-1)
        fibonacci(2.3)