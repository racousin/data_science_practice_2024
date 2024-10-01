def multiply(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        return 'error'
    
def fibonacci(n):
    if isinstance(n, int):
        if n == 0:
            return 0
        if n == 1:
            return 1
        else:
            for _ in range(n - 1):
                a, b = b, a + b
            return b
    else:
        return 'error'