
def multiply(a,b):
    M=[int, float]
    if (type(a) in M) and (type(b) in M):
    #if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        #raise ValueError("Erreur dans les arguments.")
        return a*b
    else:
        print("error")
        