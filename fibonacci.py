def fibonacci(n: int) -> int:
    """
    Generate the nth Fibonacci number using a recursive approach.

    Args:
        n (int): The position in the Fibonacci sequence (0-indexed).

    Returns:
        int: The Fibonacci number at the given position.

    Raises:
        ValueError: If the input is a negative number.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(5)
        5
    """
    # Validate input
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative indices")
    
    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Recursive case
    return fibonacci(n - 1) + fibonacci(n - 2)