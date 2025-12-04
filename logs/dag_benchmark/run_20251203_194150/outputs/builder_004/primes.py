import math

def is_prime(n: int) -> bool:
    """
    Determine if a given number is prime.

    Args:
        n (int): The number to check for primality.

    Returns:
        bool: True if the number is prime, False otherwise.

    Notes:
    - Returns False for numbers less than 2
    - Uses an efficient algorithm checking divisibility up to sqrt(n)
    """
    # Handle edge cases: numbers less than 2 are not prime
    if n < 2:
        return False
    
    # Special case for 2, which is prime
    if n == 2:
        return True
    
    # Check for even numbers greater than 2
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    for divisor in range(3, int(math.sqrt(n)) + 1, 2):
        if n % divisor == 0:
            return False
    
    # If no divisors found, the number is prime
    return True