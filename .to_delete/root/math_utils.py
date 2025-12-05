def add_numbers(a: float, b: float) -> float:
    """
    Add two numeric values and return their sum.

    Args:
        a (float): First number to be added
        b (float): Second number to be added

    Returns:
        float: The sum of a and b

    Raises:
        TypeError: If inputs are not numeric (int or float)
    """
    # Validate input types
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numeric (int or float)")
    
    # Perform addition and return result
    return a + b