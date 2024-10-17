def multiply(a, b):
    if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
        return a*b
    else:
        return "error"