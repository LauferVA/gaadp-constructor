def safe_divide(a: float, b: float) -> float:
    """
    Perform a safe division operation with explicit zero division handling.

    Args:
        a (float): The numerator of the division.
        b (float): The denominator of the division.

    Returns:
        float: The result of a divided by b.

    Raises:
        TypeError: If inputs are not numeric (int or float).
        ZeroDivisionError: If the denominator is zero.
    """
    # Validate input types
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numeric (int or float)")

    # Check for zero division
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")

    # Perform division and return as float
    return float(a / b)