
def multiply(a,b):
    M=[int, float]
    if (type(a) in M) and (type(b) in M):
        return a*b
    else:
        return "error"
        