import unittest
import warnings
import argparse
import time


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


class TimedTestResult(unittest.TextTestResult):
    """
    Custom TestResult class to report test timing and success.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_test_start_time = (
            None  # Track the start time of the first test
        )

    def startTest(self, test):
        # Record the start time of the first test
        if self._first_test_start_time is None:
            self._first_test_start_time = time.time()

        self._start_time = (
            time.time()
        )  # Record the start time of the current test
        super().startTest(test)

    def stopTest(self, test):
        elapsed_time = (
            time.time() - self._start_time
        )  # Calculate elapsed time for the current test
        total_elapsed_time = (
            time.time() - self._first_test_start_time
        )  # Time since the first test started

        # Print elapsed time for the current test and total elapsed time since the first test
        self.stream.write(
            f" ({elapsed_time:.3f}s, total: {total_elapsed_time:.3f}s)\n"
        )
        super().stopTest(test)


class TimedTestRunner(unittest.TextTestRunner):
    """
    Custom TestRunner class to use TimedTestResult.
    """

    def _makeResult(self):
        return TimedTestResult(self.stream, self.descriptions, self.verbosity)


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

    # Run the filtered tests with the custom runner
    runner = TimedTestRunner(verbosity=2)
    runner.run(filtered_suite)

# import unittest
# import warnings
# import argparse
# import time


# def collect_tests(suite, ignore_patterns):
#     """
#     Filter tests in a TestSuite based on ignore patterns.
#     """
#     filtered_suite = unittest.TestSuite()
#     for test in suite:
#         if isinstance(test, unittest.TestSuite):
#             # Recursively filter nested TestSuites
#             filtered_suite.addTests(collect_tests(test, ignore_patterns))
#         elif any(
#             pattern in test._testMethodName for pattern in ignore_patterns
#         ):
#             # Skip tests that match any ignore pattern
#             continue
#         else:
#             filtered_suite.addTest(test)
#     return filtered_suite


# class TimedTestResult(unittest.TextTestResult):
#     """
#     Custom TestResult class to report test timing and success.
#     """

#     def startTest(self, test):
#         self._start_time = time.time()  # Record the start time
#         super().startTest(test)

#     def stopTest(self, test):
#         elapsed_time = time.time() - self._start_time  # Calculate elapsed time
#         self.stream.write(f" ({elapsed_time:.3f}s)\n")  # Print elapsed time
#         super().stopTest(test)


# class TimedTestRunner(unittest.TextTestRunner):
#     """
#     Custom TestRunner class to use TimedTestResult.
#     """

#     def _makeResult(self):
#         return TimedTestResult(self.stream, self.descriptions, self.verbosity)


# if __name__ == "__main__":
#     # Show warnings but not treat them as errors
#     warnings.simplefilter("default")

#     # Set up argument parser
#     parser = argparse.ArgumentParser(
#         description="Run filtered unittest tests."
#     )
#     parser.add_argument(
#         "--start-dir",
#         type=str,
#         default="pyapprox",
#         help="Directory to start test discovery (default: pyapprox). "
#         "Use module names, e.g. pyapprox.variables",
#     )
#     parser.add_argument(
#         "--ignore-patterns",
#         type=str,
#         nargs="*",
#         default=["test_slow"],
#         help="List of patterns to ignore in test names (default: ['test_slow'])",
#     )

#     # Parse command-line arguments
#     args = parser.parse_args()

#     # Use the provided start_dir or default value
#     start_dir = args.start_dir
#     ignore_patterns = args.ignore_patterns

#     # Discover all tests in the specified directory
#     loader = unittest.TestLoader()
#     suite = loader.discover(start_dir=start_dir, pattern="test_*.py")

#     # Apply the custom filter to the discovered tests
#     filtered_suite = collect_tests(suite, ignore_patterns)

#     # Run the filtered tests with the custom runner
#     runner = TimedTestRunner(verbosity=2)
#     runner.run(filtered_suite)
