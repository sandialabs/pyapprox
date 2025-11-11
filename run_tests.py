import unittest
import argparse
import warnings


def collect_tests(suite, ignore_patterns):
    """
    Filter tests in a TestSuite based on ignore patterns.
    """
    filtered_suite = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            # Recursively filter nested TestSuites
            filtered_suite.addTests(collect_tests(test, ignore_patterns))
        elif any(
            pattern in test._testMethodName for pattern in ignore_patterns
        ):
            # Skip tests that match any ignore pattern
            continue
        else:
            filtered_suite.addTest(test)
    return filtered_suite


if __name__ == "__main__":
    # Show warnings but not treat them as errors
    warnings.simplefilter("default")

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run filtered unittest tests."
    )
    parser.add_argument(
        "--start-dir",
        type=str,
        default="pyapprox",
        help="Directory to start test discovery (default: pyapprox). "
        "Use module names, e.g. pyapprox.variables",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        nargs="*",
        default=["test_slow"],
        help="List of patterns to ignore in test names (default: ['test_slow'])",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Use the provided start_dir or default value
    start_dir = args.start_dir
    ignore_patterns = args.ignore_patterns

    # Discover all tests in the specified directory
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")

    # Apply the custom filter to the discovered tests
    filtered_suite = collect_tests(suite, ignore_patterns)

    # Run the filtered tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(filtered_suite)
