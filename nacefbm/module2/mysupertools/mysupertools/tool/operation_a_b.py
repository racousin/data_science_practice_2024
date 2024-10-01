def multiply(a, b):
    # Check if both a and b are numbers (int or float)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        return "error"

