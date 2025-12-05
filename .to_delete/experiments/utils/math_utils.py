from typing import Union, Optional

def safe_divide(numerator: Union[int, float], denominator: Union[int, float]) -> Optional[float]:
    """
    Safely divide two numbers, handling potential division by zero.

    This function performs division while preventing runtime errors from division by zero.
    If the denominator is zero, the function returns None instead of raising an exception.

    Args:
        numerator (int or float): The number to be divided.
        denominator (int or float): The number to divide by.

    Returns:
        Optional[float]: The result of the division, or None if division by zero would occur.

    Examples:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        None
    """
    if denominator == 0:
        return None
    return numerator / denominator