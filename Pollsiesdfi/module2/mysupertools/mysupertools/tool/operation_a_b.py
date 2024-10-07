def multiply(a, b):
    try:
        result = a * b
        return result
    except TypeError:
        return "error"
