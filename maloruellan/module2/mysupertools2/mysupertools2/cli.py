import argparse
from mysupertools2.math_operations import multiply, fibonacci

def main():
    parser = argparse.ArgumentParser(description = 'mysupertools2 CLI')
    parser.add_argument('operation', choices = ['multiply', 'fibonacci'], help = 'Operation')
    parser.add_argument('numbers', nargs = '+', type = int, help = 'Numbers')

    args = parser.parse_args()

    if args.operation == 'multiply':
        if len(args.numbers) != 2:
            print('Error: Multiply requires exactly 2 numbers.')
            return
        result = multiply(args.numbers[0], args.numbers[1])
        print(f'{args.numbers[0]} * {args.numbers[1]} = {result}')
    elif args.operation == 'fibonacci':
        if len(args.numbers) != 1:
            print('Error: Fibonacci requires exactly 1 number.')
            return
        result = fibonacci(args.numbers[0])
        if args.numbers[0] == 1:
            print(f'The {args.numbers[0]}st Fibonacci number is {result}.')
        if args.numbers[0] == 2:
            print(f'The {args.numbers[0]}nd Fibonacci number is {result}.')
        if args.numbers[0] == 3:
            print(f'The {args.numbers[0]}rd Fibonacci number is {result}.')
        else:
            print(f'The {args.numbers[0]}th Fibonacci number is {result}.')

if __name__ == '__main__':
    main()
