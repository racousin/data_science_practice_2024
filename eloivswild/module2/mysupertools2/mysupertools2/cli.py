import argparse
from mysupertools2.math_operations import multiply, fibonacci

def main():
    parser = argparse.ArgumentParser(description='mysupertools2 CLI')
    
 # Add arguments
    parser.add_argument('operation', choices=['multiply', 'fibonacci'], help='Choose the operation: multiply or fibonacci')
    parser.add_argument('numbers', nargs='+', type=int, help='Provide numbers for the operation')

    # Parse arguments
    args = parser.parse_args()

    # Logic to handle operations
    if args.operation == 'multiply':
        # Validate if exactly two numbers are provided
        if len(args.numbers) != 2:
            print("Error: multiply operation requires exactly two numbers.")
        else:
            # Call the multiply function with two numbers
            result = multiply(args.numbers[0], args.numbers[1])
            print(f"Result of multiply({args.numbers[0]}, {args.numbers[1]}): {result}")

    elif args.operation == 'fibonacci':
        # Validate if exactly one number is provided
        if len(args.numbers) != 1:
            print("Error: fibonacci operation requires exactly one number.")
        else:
            # Call the fibonacci function with one number
            result = fibonacci(args.numbers[0])
            print(f"Result of fibonacci({args.numbers[0]}): {result}")

if __name__ == "__main__":
    main()

