#!/usr/bin/env python3
"""
Test runner for CausalHES framework.

This script runs the test suite and provides coverage reporting.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(coverage=True, verbose=True):
    """
    Run the test suite.
    
    Args:
        coverage: Whether to run with coverage reporting
        verbose: Whether to run in verbose mode
    """
    print("🧪 Running CausalHES Test Suite")
    print("=" * 50)
    
    # Change to project directory
    project_root = Path(__file__).parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # Run tests
        result = subprocess.run(cmd, cwd=project_root, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            if coverage:
                print("📊 Coverage report generated in htmlcov/")
        else:
            print(f"\n❌ Tests failed with return code {result.returncode}")
            
        return result.returncode
        
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CausalHES tests")
    parser.add_argument(
        "--no-coverage", 
        action="store_true", 
        help="Skip coverage reporting"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="Run in quiet mode"
    )
    
    args = parser.parse_args()
    
    coverage = not args.no_coverage
    verbose = not args.quiet
    
    return run_tests(coverage=coverage, verbose=verbose)


if __name__ == "__main__":
    sys.exit(main())
