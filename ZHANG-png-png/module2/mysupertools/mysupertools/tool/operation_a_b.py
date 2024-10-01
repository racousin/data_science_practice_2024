def multiply(a, b):
    """
    Multiplies two values if they are numbers.
    Returns the product if both arguments are numbers, and 'error' otherwise.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b
    else:
        return "error"
