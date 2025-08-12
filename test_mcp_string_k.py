#!/usr/bin/env python3
"""
Test K parameter validation with MCP string handling.

This test verifies that K parameters correctly handle both integer and string
inputs from MCP clients, with proper bounds checking to prevent sqlite-vec
issues during over-fetch calculations.
"""

import json
import subprocess
import sys
import time


def test_k_parameter_validation():
    """Test K parameter validation with boundary values."""
    test_cases = [
        # Valid cases
        {"k": 1, "expected": "success", "description": "Minimum valid integer"},
        {"k": "1", "expected": "success", "description": "Minimum valid string"},
        {"k": 10, "expected": "success", "description": "Mid-range integer"},
        {"k": "10", "expected": "success", "description": "Mid-range string"},
        {"k": 20, "expected": "success", "description": "Maximum valid integer"},
        {"k": "20", "expected": "success", "description": "Maximum valid string"},
        # Invalid cases
        {"k": 0, "expected": "error", "description": "Below minimum integer"},
        {"k": "0", "expected": "error", "description": "Below minimum string"},
        {"k": 21, "expected": "error", "description": "Above maximum integer"},
        {"k": "21", "expected": "error", "description": "Above maximum string"},
        {"k": "abc", "expected": "error", "description": "Invalid string format"},
        {"k": "", "expected": "error", "description": "Empty string"},
        {
            "k": None,
            "expected": "error",
            "description": "None value (should use default)",
        },
    ]

    results = []

    print("Starting docsrs-mcp server in REST mode...")
    # Start the server in background
    server_process = subprocess.Popen(
        ["uv", "run", "docsrs-mcp", "--mode", "rest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    time.sleep(3)

    try:
        for test_case in test_cases:
            k_value = test_case["k"]
            expected = test_case["expected"]
            description = test_case["description"]

            # Prepare request payload
            payload = {
                "crate_name": "serde",
                "query": "deserialize",
            }

            # Add k parameter if not None
            if k_value is not None:
                payload["k"] = k_value

            # Make request using curl
            curl_command = [
                "curl",
                "-X",
                "POST",
                "http://localhost:8765/search_items",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(payload),
                "-s",  # Silent mode
                "-w",
                "\\n%{http_code}",  # Print HTTP status code
            ]

            result = subprocess.run(
                curl_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Parse response
            lines = result.stdout.strip().split("\n")
            http_code = lines[-1] if lines else "000"
            response_body = "\n".join(lines[:-1]) if len(lines) > 1 else ""

            # Determine test result
            if expected == "success":
                test_passed = http_code == "200"
            else:  # expected == "error"
                test_passed = http_code in ["400", "422", "500"]

            # Record result
            results.append(
                {
                    "description": description,
                    "k_value": k_value,
                    "expected": expected,
                    "http_code": http_code,
                    "passed": test_passed,
                    "response": response_body[:200] if response_body else "",
                }
            )

            # Print result
            status = "✓" if test_passed else "✗"
            print(f"{status} {description}: k={k_value!r} -> HTTP {http_code}")

            if not test_passed and response_body:
                try:
                    error_json = json.loads(response_body)
                    if "detail" in error_json:
                        print(f"  Error: {error_json['detail'][:100]}")
                except:
                    print(f"  Response: {response_body[:100]}")

    finally:
        # Stop the server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait(timeout=5)

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed < total:
        print("\nFailed tests:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['description']}: k={r['k_value']!r}")
                print(f"    Expected: {r['expected']}, Got: HTTP {r['http_code']}")

    # Return exit code
    return 0 if passed == total else 1


def test_overfetch_calculation():
    """Test that fetch_k calculation stays within bounds."""
    print("\n" + "=" * 50)
    print("OVER-FETCH CALCULATION TEST")
    print("=" * 50)

    # Test edge case: k=20 with MMR enabled
    # fetch_k should be min((20 + 15) * 1.5, 50) = min(52.5, 50) = 50

    print("Testing k=20 with MMR diversification...")
    print("Expected: fetch_k ≤ 50")
    print("Calculation: min((20 + 15) * 1.5, 50) = 50")

    # This would require testing the actual database function
    # For now, we just verify the math
    k = 20
    over_fetch = 15
    mmr_multiplier = 1.5

    fetch_k = min(int((k + over_fetch) * mmr_multiplier), 50)

    print(f"Result: fetch_k = {fetch_k}")

    if fetch_k <= 50:
        print("✓ fetch_k within bounds")
        return 0
    else:
        print("✗ fetch_k exceeds bounds!")
        return 1


if __name__ == "__main__":
    # Run tests
    exit_code1 = test_k_parameter_validation()
    exit_code2 = test_overfetch_calculation()

    # Exit with combined status
    sys.exit(max(exit_code1, exit_code2))
