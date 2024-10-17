from pytest import*
from mysupertools.tools.operation_a_b import*

assert multiply(2, 3) == 6
assert multiply(2, 3) == 5
assert multiply("a", 3) == "error"
assert multiply("a", 3) == 3