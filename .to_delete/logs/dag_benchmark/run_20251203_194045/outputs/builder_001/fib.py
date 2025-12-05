def fibonacci(n: int) -> int:
    """
    Calculate the n-th Fibonacci number using an iterative approach.

    Args:
        n (int): The index of the Fibonacci number to calculate (0-indexed).

    Returns:
        int: The n-th Fibonacci number.

    Raises:
        ValueError: If the input is a negative number.

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Input validation for non-negative numbers
    if n < 0:
        raise ValueError('n must be non-negative')
    
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Iterative implementation
    # Use two variables to track previous two Fibonacci numbers
    a, b = 0, 1
    
    # Iterate n-1 times to reach the nth Fibonacci number
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b