"""
Location Geocoder
Maps location names from agent reports to latitude/longitude coordinates.
Uses a lookup table for common Tanzanian and East African locations.
"""

from typing import Optional, Tuple, List
import re


# Location lookup table for common Tanzanian and East African locations
LOCATION_LOOKUP = {
    # Major cities
    "dar es salaam": (-6.7924, 39.2083),
    "dar": (-6.7924, 39.2083),
    "nairobi": (-1.2921, 36.8219),
    "mombasa": (-4.0435, 39.6682),
    "kisumu": (-0.0917, 34.7680),
    "arusha": (-3.3869, 36.6830),
    "dodoma": (-6.1630, 35.7516),
    "kampala": (0.3476, 32.5825),
    "kigali": (-1.9441, 30.0619),
    "tanga": (-5.0695, 39.0996),
    "morogoro": (-6.8167, 37.6667),
    "zanzibar": (-6.1659, 39.2026),
    "kimara": (-6.7500, 39.2000),  # Suburb of Dar es Salaam
    
    # Hospitals and medical facilities
    "hospitali ya muhimbili": (-6.7894, 39.2083),
    "muhimbili": (-6.7894, 39.2083),
    "muhimbili hospital": (-6.7894, 39.2083),
    
    # Universities and educational institutions
    "chuo kikuu cha nairobi": (-1.2921, 36.8219),
    "university of nairobi": (-1.2921, 36.8219),
    "chuo kikuu": (-1.2921, 36.8219),  # Default to Nairobi if just "university"
    
    # Common location keywords (will use city center)
    "soko": (-6.7924, 39.2083),  # Market - default to Dar
    "shule": (-6.7924, 39.2083),  # School - default to Dar
    "stesheni": (-6.7924, 39.2083),  # Station - default to Dar
    "kituo": (-6.7924, 39.2083),  # Center/Station - default to Dar
    "ofisi": (-6.7924, 39.2083),  # Office - default to Dar
    "jengo": (-6.7924, 39.2083),  # Building - default to Dar
    "nyumba": (-6.7924, 39.2083),  # House - default to Dar
}

# Default location (Dar es Salaam center)
DEFAULT_LOCATION = (-6.7924, 39.2083)


def normalize_location_name(location: str) -> str:
    """
    Normalize location name for lookup.
    
    Args:
        location: Location name string
        
    Returns:
        Normalized location name (lowercase, trimmed)
    """
    # Remove common prefixes and clean up
    location = location.lower().strip()
    location = re.sub(r'^(katika|kwenye|eneo|mahali|la|ya|cha)\s+', '', location)
    location = re.sub(r'\s+', ' ', location)  # Normalize whitespace
    return location.strip()


def geocode_location(location_name: str) -> Tuple[float, float]:
    """
    Geocode a location name to lat/lon coordinates.
    
    Args:
        location_name: Location name from agent report
        
    Returns:
        Tuple of (latitude, longitude)
    """
    if not location_name:
        return DEFAULT_LOCATION
    
    # Normalize the location name
    normalized = normalize_location_name(location_name)
    
    # Direct lookup
    if normalized in LOCATION_LOOKUP:
        return LOCATION_LOOKUP[normalized]
    
    # Try partial matches (e.g., "muhimbili" matches "hospitali ya muhimbili")
    for key, coords in LOCATION_LOOKUP.items():
        if normalized in key or key in normalized:
            return coords
    
    # If no match found, return default (Dar es Salaam)
    return DEFAULT_LOCATION


def extract_location_from_report(report: dict) -> Optional[str]:
    """
    Extract location string from agent report.
    
    Args:
        report: Agent report dictionary
        
    Returns:
        Location string or None
    """
    try:
        # Try to get location from validation.entities.where
        validation = report.get('validation', {})
        entities = validation.get('entities', {})
        where_list = entities.get('where', [])
        
        if where_list and len(where_list) > 0:
            # Use first location found
            return where_list[0]
        
        # Fallback: try to extract from raw_message using simple patterns
        raw_message = report.get('raw_message', '')
        # Look for common location patterns
        location_patterns = [
            r'(?:katika|kwenye|eneo|mahali)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(dar es salaam|nairobi|mombasa|arusha|dodoma|kampala|kigali)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, raw_message, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    except Exception:
        return None


def geocode_report_location(report: dict) -> Tuple[float, float]:
    """
    Extract and geocode location from an agent report.
    
    Args:
        report: Agent report dictionary
        
    Returns:
        Tuple of (latitude, longitude)
    """
    location_name = extract_location_from_report(report)
    if location_name:
        return geocode_location(location_name)
    return DEFAULT_LOCATION

