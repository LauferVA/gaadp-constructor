def binary_search(arr: list[int], target: int) -> int:
    """
    Perform binary search to find the index of target in a sorted array.

    Args:
        arr (list[int]): A sorted array of integers in ascending order
        target (int): The value to search for in the array

    Returns:
        int: Index of the target if found, -1 otherwise

    Time Complexity: O(log n)
    Space Complexity: O(1)

    Raises:
        TypeError: If input types are incorrect
    """
    # Type checking
    if not isinstance(arr, list):
        raise TypeError("Input must be a list")
    if not isinstance(target, int):
        raise TypeError("Target must be an integer")
    
    # Handle empty array case
    if not arr:
        return -1
    
    # Iterative binary search
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Calculate middle index to avoid potential integer overflow
        mid = left + (right - left) // 2
        
        # Check if target is found
        if arr[mid] == target:
            return mid
        
        # Decide which half to search
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # Target not found
    return -1