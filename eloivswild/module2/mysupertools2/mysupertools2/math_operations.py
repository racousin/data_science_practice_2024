def multiply(a, b):
    try:
        result = a * b
        return result
    except TypeError:
        return "error"
    
def fibonacci(n):
    if n <= 0:
        return "Invalid input, n must be a positive integer"
    elif n == 1:
        return 0  # The first Fibonacci number is 0
    elif n == 2:
        return 1  # The second Fibonacci number is 1

    # Iterative approach to calculate nth Fibonacci number
    a, b = 0, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b