def multiply(a, b):
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return a*b
    else : 
        return "error"