"""
Task Scheduler Module
---------------------
This module provides an efficient `TaskScheduler` class for managing and executing tasks asynchronously using
a thread pool. It leverages the Python `concurrent.futures` library to handle multiple tasks, supporting
asynchronous execution with error handling and custom timeout settings. This utility is well-suited for
high-performance scenarios where numerous tasks need to be processed concurrently, with options to specify
the number of workers and handle task failures gracefully.

Author: Anand Kamble
Date: July 22, 2024
Usage: Designed for automated task execution and timing in multi-threaded environments, ideal for inclusion in 
       large-scale, resource-intensive applications like automated grading systems or data processing pipelines.
"""

from typing import List, Callable, Any
import concurrent.futures
from timer import PerfCounterTimer
import multiprocessing


class TaskScheduler:
    """
    A TaskScheduler for handling and executing multiple tasks concurrently using a thread pool.
    
    Attributes:
    ----------
    max_workers : int
        Maximum number of worker threads for concurrent execution.
    timeout : int
        Timeout for waiting for all tasks to complete (in seconds).
    executor : concurrent.futures.ThreadPoolExecutor
        The ThreadPoolExecutor instance to manage worker threads.
    tasks : List[Tuple[Callable[..., Any], tuple]]
        List of all tasks added with their arguments.
    failed_tasks : List[Tuple[Callable[..., Any], tuple]]
        List of tasks that failed to execute.
    futures : List[concurrent.futures.Future]
        List of Future objects associated with each task.
    timer : PerfCounterTimer
        A timer instance to monitor task execution times.

    Methods:
    -------
    add_task(id: str, task: Callable[..., Any], *args)
        Adds a new task to the scheduler and submits it to the thread pool for execution.
    execute_tasks()
        Waits for all tasks to complete or for the timeout to elapse.
    get_results() -> Tuple[List[Any], List[Tuple[Callable[..., Any], tuple]]]
        Retrieves the results of completed tasks and returns them alongside any failed tasks.
    """

    def __init__(self, max_workers: int = multiprocessing.cpu_count(), timeout: int = 60):
        """
        Initializes the TaskScheduler with a specified number of worker threads and a timeout period.

        Parameters:
        ----------
        max_workers : int, optional
            The maximum number of worker threads to use for concurrent execution (default is the CPU core count).
        timeout : int, optional
            The time (in seconds) to wait for tasks to complete (default is 60 seconds).
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )
        self.tasks = []
        self.failed_tasks = []
        self.futures = []
        self.timer = PerfCounterTimer()
        self.timeout = timeout

    def add_task(self, id: str, task: Callable[..., Any], *args):
        """
        Adds a new task to the scheduler, submits it to the executor, and starts timing the task execution.

        Parameters:
        ----------
        id : str
            Unique identifier for the task, used for timing and tracking.
        task : Callable[..., Any]
            The function or callable to be executed as a task.
        *args
            Positional arguments to pass to the task function.
        """
        with PerfCounterTimer(id).timeit():
            future = self.executor.submit(task, *args)
            self.futures.append(future)
            self.tasks.append((task, args))
            future.add_done_callback(self._task_done)

    def _task_done(self, future):
        """
        Callback method invoked upon task completion, handling both success and error scenarios.

        Parameters:
        ----------
        future : concurrent.futures.Future
            The Future object associated with the completed task.
        """
        try:
            result = future.result()
            print(f"Task completed with result: {result}")
        except Exception as e:
            print(f"An error occurred: {e}")
            self.failed_tasks.append(self.tasks[self.futures.index(future)])

    def execute_tasks(self):
        """
        Blocks execution until all tasks are complete or the timeout period is reached.
        """
        concurrent.futures.wait(self.futures, timeout=self.timeout)

    def get_results(self) -> tuple[List[Any], List[tuple[Callable[..., Any], tuple]]]:
        """
        Gathers results from all completed tasks and logs any tasks that failed.

        Returns:
        -------
        tuple[List[Any], List[Tuple[Callable[..., Any], tuple]]]
            A tuple containing a list of task results and a list of failed tasks.
        """
        results = []
        for future in self.futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"An error occurred: {e}")
                self.failed_tasks.append(
                    self.tasks[self.futures.index(future)]
                )
        return results, self.failed_tasks
