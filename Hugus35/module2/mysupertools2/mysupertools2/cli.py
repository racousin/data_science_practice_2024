import argparse
from mysupertools2.math_operations import multiply, fibonacci


def main():

    fonctions  = {
        'multiply' : multiply,
        'fibonacci' : fibonacci
    }

    parser = argparse.ArgumentParser(description='mysupertools2 CLI')
    
    # Add arguments here
    parser.add_argument("operation", choices=['multiply', 'fibonacci'])
    parser.add_argument("a", type = int)
    parser.add_argument("b", type = int, nargs = '?')

    args = parser.parse_args()

    if (args.operation == 'multiply') and (args.b != None):
        print(f"{args.a}*{args.b} = {multiply(args.a, args.b)}")

    elif (args.operation == 'fibonacci') and (args.b == None):
        print(f"Le {args.a}ème nombre de Fibonacci est = {fibonacci(args.a)}")
    
    else : 
        print("Erreur : arguments ou nom de fonction non-valides")

if __name__ == '__main__':
    main()