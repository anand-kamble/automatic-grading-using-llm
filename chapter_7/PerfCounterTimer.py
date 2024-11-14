"""
Performance Counter Timer Module
--------------------------------
This module provides the `PerfCounterTimer` class for precise timing of code segments, tracking performance metrics,
and organizing data into a DataFrame for easy analysis. This is particularly useful in performance-critical applications
such as benchmarking tasks in large-scale automated systems.

The `PerfCounterTimer` class supports multiple timers, allowing for easy comparison of execution times for different
code blocks or functions. Each timer captures statistics including minimum, mean, and standard deviation of execution times,
as well as the number of times the code block was executed. The module also supports detailed reporting and logging to
a DataFrame, enabling persistent and structured data storage for further analysis.

Author: Gordon Erlebacher
Date: July 17, 2024
Usage: Ideal for automated task execution timing and performance analysis in concurrent and multi-threaded environments.
"""

import time
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import pandas as pd


class PerfCounterTimer:
    """
    A high-precision timer that tracks execution times of code blocks, providing statistical analysis.
    
    Attributes:
    ----------
    timings : defaultdict(list)
        Stores individual timings for each timer name.
    columns : list
        Column names for the DataFrame storing the timing statistics.
    df : pd.DataFrame
        DataFrame for structured storage of timing statistics across multiple runs.

    Methods:
    -------
    timeit(count: int = 1)
        Context manager to measure execution time of a code block.
    reset()
        Resets all recorded timings.
    report(msg: str = "")
        Prints a detailed report of all recorded timings and updates the DataFrame.
    get_dataframe() -> pd.DataFrame
        Returns the DataFrame containing all timing statistics.
    """

    timings = defaultdict(list)
    columns = ["name", "min", "mean", "std", "count"]
    df = pd.DataFrame(columns=columns)

    def __init__(self, name=""):
        """
        Initializes the PerfCounterTimer with a specific name.

        Parameters:
        ----------
        name : str, optional
            The name identifier for the timer (default is an empty string).
        """
        self.name = name

    @contextmanager
    def timeit(self, count: int = 1):
        """
        Context manager to time the execution of a code block and record its duration.

        Parameters:
        ----------
        count : int, optional
            Number of repetitions for averaging the execution time (default is 1).
        """
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) / count
        PerfCounterTimer.timings[self.name].append(elapsed_time)

    @classmethod
    def reset(cls):
        """Resets all recorded timings across all timers."""
        cls.timings = defaultdict(list)

    def __call__(self, count: int = 1):
        """
        Allows the PerfCounterTimer to be used as a callable, returning the context manager.

        Parameters:
        ----------
        count : int, optional
            Number of repetitions for averaging the execution time (default is 1).
        """
        return self.timeit(count)

    @classmethod
    def report(cls, msg="") -> defaultdict:
        """
        Generates a report of all recorded timings, including mean, standard deviation, and minimum time.
        Also updates the DataFrame with the collected timing statistics.

        Parameters:
        ----------
        msg : str, optional
            An optional message to display with the report (default is an empty string).

        Returns:
        -------
        defaultdict
            A dictionary containing timing statistics for each timer name.
        """
        out_dict = defaultdict(dict)
        if msg:
            print(f"\n{msg}")
        for name, times in cls.timings.items():
            out_dict[name] = {}
            mean_total_time = np.mean(times)
            std_total_time = np.std(times)
            min_total_time = np.min(times)
            counter = len(times)
            out_dict[name]["mean"] = mean_total_time
            out_dict[name]["std"] = std_total_time
            out_dict[name]["min"] = min_total_time
            out_dict[name]["count"] = counter
            print(
                f"Name: {name}, Count: {counter}, Total Time: {min_total_time:7.4f} seconds, "
                f"Timings: {mean_total_time / counter:7.4f} each"
            )
        print()

        # Update the DataFrame with the new data
        data_dict = defaultdict(dict)
        for name, times in cls.timings.items():
            mean_total_time = np.mean(times)
            std_total_time = np.std(times)
            min_total_time = np.min(times)
            counter = len(times)
            data_dict[name]["name"] = name
            data_dict[name]["mean"] = mean_total_time
            data_dict[name]["std"] = std_total_time
            data_dict[name]["min"] = min_total_time
            data_dict[name]["count"] = counter

        # Convert data_dict to DataFrame and concatenate with existing df
        new_rows = pd.DataFrame.from_dict(data_dict, orient="index")
        cls.df = pd.concat([cls.df, new_rows], ignore_index=True)

    @classmethod
    def get_dataframe(cls) -> pd.DataFrame:
        """
        Returns the DataFrame containing all timing statistics.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with timing data for each timer name.
        """
        return cls.df


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    timer = PerfCounterTimer()

    # Simulating multiple timed blocks
    with PerfCounterTimer("gordon").timeit():
        time.sleep(0.5)
    with PerfCounterTimer("gordon").timeit():
        time.sleep(0.5)

    with PerfCounterTimer("frances").timeit():
        time.sleep(0.7)
    with PerfCounterTimer("frances").timeit():
        time.sleep(0.7)

    with PerfCounterTimer("ggordon").timeit():
        time.sleep(0.3)
    with PerfCounterTimer("ggordon").timeit():
        time.sleep(0.3)

    # Print the report
    PerfCounterTimer.report()
