"""
DateTime Utilities

Provides modern, timezone-aware datetime utilities to replace deprecated methods.
All functions use the recommended timezone-aware datetime.now(timezone.utc) 
instead of deprecated datetime.utcnow() and datetime.utcfromtimestamp().
"""

from datetime import datetime, timezone
from typing import Union, Optional


def current_utc() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def current_utc_str(format: str = "%Y-%m-%dT%H:%M:%S.%fZ") -> str:
    """Get current UTC datetime as formatted string."""
    return current_utc().strftime(format)


def current_utc_int() -> int:
    return int(current_utc().timestamp())


def current_utc_timestamp() -> int:
    """Get current UTC timestamp as integer (Unix epoch seconds)."""
    return int(current_utc().timestamp())


def datetime_from_timestamp(timestamp: Union[int, float]) -> datetime:
    """
    Create timezone-aware datetime from Unix timestamp.
    
    Replaces deprecated datetime.utcfromtimestamp().
    
    Args:
        timestamp: Unix timestamp (seconds since epoch)
        
    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: Datetime object (timezone-aware or naive treated as UTC)
        
    Returns:
        Unix timestamp as integer
    """
    if dt.tzinfo is None:
        # Treat naive datetime as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def datetime_to_iso_string(dt: datetime, with_microseconds: bool = True) -> str:
    """
    Convert datetime to ISO 8601 string with Z suffix.
    
    Args:
        dt: Datetime object
        with_microseconds: Whether to include microseconds
        
    Returns:
        ISO 8601 string ending with 'Z'
    """
    if dt.tzinfo is None:
        # Treat naive datetime as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Convert to UTC if not already
    if dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)
    
    if with_microseconds:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def is_future_datetime(dt: datetime, reference: Optional[datetime] = None) -> bool:
    """
    Check if datetime is in the future.
    
    Args:
        dt: Datetime to check
        reference: Reference datetime (default: current UTC)
        
    Returns:
        True if dt is in the future relative to reference
    """
    if reference is None:
        reference = current_utc()
    
    # Ensure both datetimes are timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    
    return dt > reference


def add_minutes_to_datetime(dt: datetime, minutes: int) -> datetime:
    """
    Add minutes to a datetime.
    
    Args:
        dt: Base datetime
        minutes: Minutes to add
        
    Returns:
        New datetime with minutes added
    """
    from datetime import timedelta
    return dt + timedelta(minutes=minutes)


def add_days_to_datetime(dt: datetime, days: int) -> datetime:
    """
    Add days to a datetime.
    
    Args:
        dt: Base datetime
        days: Days to add
        
    Returns:
        New datetime with days added
    """
    from datetime import timedelta
    return dt + timedelta(days=days)


def format_duration_ms(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    """
    Calculate duration in milliseconds between two datetimes.
    
    Args:
        start_time: Start datetime
        end_time: End datetime (default: current UTC)
        
    Returns:
        Duration in milliseconds
    """
    if end_time is None:
        end_time = current_utc()
    
    # Ensure both are timezone-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    delta = end_time - start_time
    return delta.total_seconds() * 1000
