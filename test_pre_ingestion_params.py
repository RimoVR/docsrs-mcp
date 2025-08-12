#!/usr/bin/env python3
"""Test StartPreIngestionRequest parameter validation."""

import pytest
from pydantic import ValidationError

from src.docsrs_mcp.models import StartPreIngestionRequest


def test_force_parameter_boolean_coercion():
    """Test that force parameter handles various boolean representations."""
    # String true values
    true_values = [
        "true",
        "True",
        "TRUE",
        "1",
        "yes",
        "Yes",
        "YES",
        "on",
        "On",
        "ON",
        "t",
        "T",
    ]
    for val in true_values:
        req = StartPreIngestionRequest(force=val)
        assert req.force is True, f"Failed for value: {val}"

    # String false values
    false_values = [
        "false",
        "False",
        "FALSE",
        "0",
        "no",
        "No",
        "NO",
        "off",
        "Off",
        "OFF",
        "f",
        "F",
        "",
    ]
    for val in false_values:
        req = StartPreIngestionRequest(force=val)
        assert req.force is False, f"Failed for value: {val}"

    # Boolean values
    req = StartPreIngestionRequest(force=True)
    assert req.force is True

    req = StartPreIngestionRequest(force=False)
    assert req.force is False

    # Integer values
    req = StartPreIngestionRequest(force=1)
    assert req.force is True

    req = StartPreIngestionRequest(force=0)
    assert req.force is False

    # Default value
    req = StartPreIngestionRequest()
    assert req.force is False


def test_concurrency_parameter_validation():
    """Test concurrency parameter with bounds checking."""
    # Valid values as strings
    valid_str_values = [("1", 1), ("3", 3), ("5", 5), ("10", 10)]
    for str_val, expected in valid_str_values:
        req = StartPreIngestionRequest(concurrency=str_val)
        assert req.concurrency == expected, f"Failed for value: {str_val}"

    # Valid values as integers
    for i in range(1, 11):
        req = StartPreIngestionRequest(concurrency=i)
        assert req.concurrency == i

    # Boundary violations (should fail)
    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(concurrency="0")
    assert "between 1 and 10" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(concurrency="11")
    assert "between 1 and 10" in str(exc_info.value)

    # Invalid strings (should fail)
    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(concurrency="abc")
    assert "concurrency" in str(exc_info.value)

    # None value (should be allowed)
    req = StartPreIngestionRequest(concurrency=None)
    assert req.concurrency is None

    # Default (not specified)
    req = StartPreIngestionRequest()
    assert req.concurrency is None


def test_count_parameter_validation():
    """Test count parameter with bounds checking."""
    # Valid values as strings
    valid_str_values = [
        ("10", 10),
        ("50", 50),
        ("100", 100),
        ("250", 250),
        ("500", 500),
    ]
    for str_val, expected in valid_str_values:
        req = StartPreIngestionRequest(count=str_val)
        assert req.count == expected, f"Failed for value: {str_val}"

    # Valid values as integers
    for val in [10, 50, 100, 250, 500]:
        req = StartPreIngestionRequest(count=val)
        assert req.count == val

    # Boundary violations (should fail)
    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(count="9")
    assert "between 10 and 500" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(count="501")
    assert "between 10 and 500" in str(exc_info.value)

    # Invalid strings (should fail)
    with pytest.raises(ValidationError) as exc_info:
        StartPreIngestionRequest(count="fifty")
    assert "count" in str(exc_info.value)

    # None value (should be allowed)
    req = StartPreIngestionRequest(count=None)
    assert req.count is None

    # Default (not specified)
    req = StartPreIngestionRequest()
    assert req.count is None


def test_all_parameters_combined():
    """Test all parameters together."""
    # All as strings (typical MCP scenario)
    req = StartPreIngestionRequest(force="true", concurrency="3", count="100")
    assert req.force is True
    assert req.concurrency == 3
    assert req.count == 100

    # Mixed types
    req = StartPreIngestionRequest(force=True, concurrency="5", count=200)
    assert req.force is True
    assert req.concurrency == 5
    assert req.count == 200

    # Only force specified
    req = StartPreIngestionRequest(force="yes")
    assert req.force is True
    assert req.concurrency is None
    assert req.count is None

    # Empty strings and null-like values
    req = StartPreIngestionRequest(force="", concurrency=None, count=None)
    assert req.force is False
    assert req.concurrency is None
    assert req.count is None


def test_edge_cases():
    """Test edge cases for parameter handling."""
    # Whitespace handling
    req = StartPreIngestionRequest(
        force="  true  ", concurrency="  5  ", count="  100  "
    )
    assert req.force is True
    assert req.concurrency == 5
    assert req.count == 100

    # Empty string for numeric parameters (should fail)
    with pytest.raises(ValidationError):
        StartPreIngestionRequest(concurrency="")

    with pytest.raises(ValidationError):
        StartPreIngestionRequest(count="")

    # Float strings (should fail for integer parameters)
    with pytest.raises(ValidationError):
        StartPreIngestionRequest(concurrency="3.5")

    with pytest.raises(ValidationError):
        StartPreIngestionRequest(count="100.5")


def test_mcp_realistic_scenarios():
    """Test realistic MCP client scenarios."""
    # Scenario 1: Force restart with default settings
    params = {"force": "true"}
    req = StartPreIngestionRequest(**params)
    assert req.force is True
    assert req.concurrency is None
    assert req.count is None

    # Scenario 2: Custom ingestion settings
    params = {"force": "false", "concurrency": "5", "count": "250"}
    req = StartPreIngestionRequest(**params)
    assert req.force is False
    assert req.concurrency == 5
    assert req.count == 250

    # Scenario 3: Empty/default request
    params = {}
    req = StartPreIngestionRequest(**params)
    assert req.force is False
    assert req.concurrency is None
    assert req.count is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
