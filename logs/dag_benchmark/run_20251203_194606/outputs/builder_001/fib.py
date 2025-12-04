def fibonacci(n: int) -> int:
    """
    Calculate the n-th Fibonacci number using an iterative approach.

    Args:
        n (int): The index of the Fibonacci number to calculate (0-indexed).

    Returns:
        int: The n-th Fibonacci number.

    Raises:
        ValueError: If the input is negative.

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Input validation for negative numbers
    if n < 0:
        raise ValueError("n must be non-negative")
    
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Iterative Fibonacci calculation
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b