import re
from datetime import datetime
from typing import Union, List

class LogValidationError(Exception):
    """Custom exception for log validation errors"""
    pass

def validate_log_level(level: str, allowed_levels: List[str] = None) -> bool:
    """
    Validate the log level against allowed levels.
    
    Args:
        level (str): Log level to validate
        allowed_levels (List[str], optional): List of allowed log levels. 
                                              Defaults to standard levels.
    
    Returns:
        bool: True if log level is valid, False otherwise
    
    Raises:
        LogValidationError if level is invalid
    """
    if allowed_levels is None:
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    if level.upper() not in allowed_levels:
        raise LogValidationError(f"Invalid log level. Allowed levels are: {allowed_levels}")
    
    return True

def validate_log_message(message: str, min_length: int = 1, max_length: int = 1000) -> bool:
    """
    Validate log message length and basic format.
    
    Args:
        message (str): Log message to validate
        min_length (int, optional): Minimum message length. Defaults to 1.
        max_length (int, optional): Maximum message length. Defaults to 1000.
    
    Returns:
        bool: True if message is valid
    
    Raises:
        LogValidationError if message is invalid
    """
    if not isinstance(message, str):
        raise LogValidationError("Log message must be a string")
    
    if len(message.strip()) < min_length:
        raise LogValidationError(f"Log message too short. Minimum length is {min_length}")
    
    if len(message) > max_length:
        raise LogValidationError(f"Log message too long. Maximum length is {max_length}")
    
    return True

def validate_log_timestamp(timestamp: Union[str, datetime], 
                            format: str = '%Y-%m-%d %H:%M:%S') -> bool:
    """
    Validate log timestamp format.
    
    Args:
        timestamp (Union[str, datetime]): Timestamp to validate
        format (str, optional): Expected timestamp format. 
                                Defaults to 'YYYY-MM-DD HH:MM:SS'
    
    Returns:
        bool: True if timestamp is valid
    
    Raises:
        LogValidationError if timestamp is invalid
    """
    if isinstance(timestamp, str):
        try:
            datetime.strptime(timestamp, format)
        except ValueError:
            raise LogValidationError(f"Invalid timestamp format. Expected {format}")
    
    elif isinstance(timestamp, datetime):
        # If it's already a datetime object, it's valid
        return True
    
    else:
        raise LogValidationError("Timestamp must be a string or datetime object")
    
    return True

def validate_log_severity(severity: int, min_severity: int = 0, max_severity: int = 5) -> bool:
    """
    Validate log severity level.
    
    Args:
        severity (int): Severity level to validate
        min_severity (int, optional): Minimum allowed severity. Defaults to 0.
        max_severity (int, optional): Maximum allowed severity. Defaults to 5.
    
    Returns:
        bool: True if severity is valid
    
    Raises:
        LogValidationError if severity is invalid
    """
    if not isinstance(severity, int):
        raise LogValidationError("Severity must be an integer")
    
    if severity < min_severity or severity > max_severity:
        raise LogValidationError(
            f"Severity out of range. Must be between {min_severity} and {max_severity}"
        )
    
    return True

def validate_log_entry(log_entry: dict) -> bool:
    """
    Comprehensive log entry validation.
    
    Args:
        log_entry (dict): Complete log entry to validate
    
    Returns:
        bool: True if entire log entry is valid
    
    Raises:
        LogValidationError for any validation failures
    """
    required_keys = ['level', 'message', 'timestamp']
    
    # Check for required keys
    for key in required_keys:
        if key not in log_entry:
            raise LogValidationError(f"Missing required log entry key: {key}")
    
    # Validate individual components
    validate_log_level(log_entry['level'])
    validate_log_message(log_entry['message'])
    validate_log_timestamp(log_entry['timestamp'])
    
    # Optional severity validation if present
    if 'severity' in log_entry:
        validate_log_severity(log_entry['severity'])
    
    return True
