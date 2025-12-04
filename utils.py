from datetime import datetime

def calculate_age(birth_year: int) -> int:
    """
    Calculate the current age based on the birth year.

    Args:
        birth_year (int): The year of birth.

    Returns:
        int: The current age calculated by subtracting birth year from current year.

    Raises:
        ValueError: If birth year is in the future or unreasonably old.
    """
    # Get the current year
    current_year = datetime.now().year

    # Validate birth year
    if birth_year > current_year:
        raise ValueError("Birth year cannot be in the future.")
    
    if birth_year < current_year - 150:
        raise ValueError("Birth year seems unrealistically old.")

    # Calculate and return age
    return current_year - birth_year