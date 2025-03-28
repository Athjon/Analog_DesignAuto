import datetime
import os
import random
import shutil
import time
import logging
import functools


def unit_conversion(value):
    """
    Convert unit-suffixed values to float.
    Handles units from micro (u) to giga (G).
    """

    if isinstance(value, int):
        return float(value)

    unit_mapping = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, "M": 1e6, 'G': 1e9
    }
    for unit, multiplier in unit_mapping.items():
        if isinstance(value, str) and value.endswith(unit):
            return float(value.replace(unit, '')) * multiplier

    return float(value)


def find_closest_value_index(target_list, target_value):
    closest_index = min(range(len(target_list)), key=lambda i: abs(target_list[i] - target_value))
    return closest_index


# Test Code
# test_values = ['0.5u', '10M', '2k', 5, '7']
# converted_values = [unit_conversion(value) for value in test_values]
# print(converted_values)

# Output
# [5e-07, 10000000.0, 1000000000.0, 100.0, 2000.0]

def create_work_dir(base_path, suffix=None):
    """
    Create a new directory for the work.
    :param base_path: base path of the work directory
    :param suffix: suffix to be added to the directory name
    :return: work_dir: path of the work directory
    """
    cur_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    random_num = str(random.randint(1000, 9999))
    # Get the thread id
    thread_id = str(os.getpid())
    dir_name = f"tmp_{cur_time}{thread_id}{random_num}_{suffix}" if suffix else f"tmp_{cur_time}{thread_id}{random_num}"
    work_dir = os.path.join(base_path, dir_name)
    os.makedirs(work_dir, exist_ok=True)
    logging.info(f"Created working directory for {suffix}: {work_dir}")
    return work_dir

def create_corner_work_dir(tt_folder_path, suffix=None):
    """
    Create a new directory for the corner work.
    :param tt_folder_path: full path of the work directory
    :param suffix: suffix to be added to the directory name
    :return: work_dir: path of the work directory
    """
    # Replace '_tt' in the folder name with '_suffix'
    corner_folder_path = tt_folder_path.replace('_tt', f'_{suffix}')
    os.makedirs(corner_folder_path, exist_ok=True)
    logging.info(f"Created corner working directory for corner {suffix}: {corner_folder_path}")
    return corner_folder_path

def retry_decorator(retry_count=2, delay_seconds=1, default_value=None, timeout_minutes=5):
    """
    A decorator for retrying a function with timeout mechanism.
    Args:
        retry_count: Maximum number of retry attempts
        delay_seconds: Delay between retries in seconds
        default_value: Default value to return if all retries fail
        timeout_minutes: Maximum execution time in minutes before returning default value
    Returns:
        Wrapped function that implements retry logic with timeout
    """

    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            nonlocal retry_count, delay_seconds
            attempts = 0
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60

            while attempts <= retry_count:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    current_time = time.time()
                    elapsed_time = current_time - start_time

                    # Check for timeout
                    if elapsed_time > timeout_seconds:
                        logging.debug(f"Function {func.__name__} exceeded timeout of {timeout_minutes} minutes.")
                        logging.debug(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
                        logging.debug(f"Returning default value: {default_value}")
                        return default_value

                    logging.debug(f"Attempt {attempts + 1} failed for {func.__name__}")
                    logging.debug(f"Error message: {str(e)}")
                    logging.debug(f"Elapsed time: {elapsed_time / 60:.2f} minutes")

                    if attempts == retry_count:
                        logging.debug(f"All {retry_count + 1} attempts failed for {func.__name__}")
                        logging.debug(f"Total execution time: {elapsed_time / 60:.2f} minutes")
                        logging.debug(f"Returning default value: {default_value}")
                        return default_value

                    print(f"Attempt {attempts + 1} failed for {func.__name__} due to {e}, "
                          f"retrying after {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                    attempts += 1

        return wrapper_retry

    return decorator_retry

# Example of usage:
# @retry_decorator(retry_count=2, delay_seconds=5, default_value=self.zero_sim_result)
# def run_spectre_simulation(working_dir, sim_config, sim_output_enable):
#     ...
#     return sim_result


def delete_work_dir(work_dir):
    """
    Delete the work directory and all its contents.
    :param work_dir: path of the work directory
    """
    try:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
    except OSError as e:
        print(f"Warning!!!: {e.strerror}. Directory {work_dir} does not exist or cannot be removed.")
        pass


def find_psr_turning_point(values, window_size=3, threshold=0.1):
    """
    Find the first turning point from decreasing to increasing in PSR trace.

    Args:
        values: List of PSR values
        window_size: Size of sliding window to determine trend
        threshold: Minimum change threshold to filter noise

    Returns:
        index: Index of turning point, or -1 if not found
    """
    if len(values) < window_size * 2:
        return -1

    # Calculate trends using sliding window
    for i in range(window_size, len(values) - window_size):
        # Check if previous window is decreasing
        prev_decreasing = all(values[j - 1] - values[j] >= threshold
                              for j in range(i - window_size + 1, i))

        # Check if next window is increasing
        next_increasing = all(values[j + 1] - values[j] >= threshold
                              for j in range(i, i + window_size - 1))

        if prev_decreasing and next_increasing:
            return i

    return -1
