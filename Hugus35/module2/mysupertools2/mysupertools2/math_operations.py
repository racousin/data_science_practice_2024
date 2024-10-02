def multiply(a, b):
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return a*b
    else : 
        return "error"
    
def fibonacci(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else : 
        return fibonacci(n-1) + fibonacci(n-2)