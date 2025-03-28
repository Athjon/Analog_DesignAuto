import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import os
import json
import logging

# from extract_trace import extractTransTrace_psf
from util.extract_trace import extractTransTrace_psf


def split_signal(data: pd.DataFrame,
                 high_threshold: float = 6.0,
                 plot_segments: int = 10,
                 output_dir: str = None,
                 debug: bool = False) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Split time series signal into segments based on high voltage regions.

    Args:
        data: DataFrame with 'time' and 'signal' columns
        high_threshold: Voltage threshold for detecting high regions
        plot_segments: Number of segments to plot for debug
        output_dir: Directory to save debug plots
        debug: Enable diagnostic outputs (default: False)

    Returns:
        Dict with segment index as key and dict of time/signal arrays as value
    """
    # Find high voltage regions
    high_voltage_mask = data.iloc[:, 1] > high_threshold

    # Find start and end indices of high voltage regions
    region_changes = np.diff(high_voltage_mask.astype(int))
    rise_indices = np.where(region_changes == 1)[0]
    fall_indices = np.where(region_changes == -1)[0]

    # Ensure equal number of rise and fall indices
    min_length = min(len(rise_indices), len(fall_indices))
    rise_indices = rise_indices[:min_length]
    fall_indices = fall_indices[:min_length]

    # Adjust indices to remove incomplete cycles
    if len(rise_indices) > 2:
        rise_indices = rise_indices[1:-1]
        fall_indices = fall_indices[1:-1]
    else:
        raise ValueError("Not enough complete signal cycles found in data")

    # Calculate midpoints between regions
    split_times = []
    for i in range(len(rise_indices) - 1):
        mid_time = (data.iloc[fall_indices[i]].time +
                    data.iloc[rise_indices[i + 1]].time) / 2
        split_times.append(mid_time)

    # Create segments dictionary
    segments = {}
    time_col = data.columns[0]
    signal_col = data.columns[1]

    for i in range(len(split_times) - 1):
        mask = (data[time_col] >= split_times[i]) & (data[time_col] < split_times[i + 1])
        segment_data = data[mask]
        segments[i] = {
            'time': segment_data[time_col].values,
            'signal': segment_data[signal_col].values
        }

    if debug:
        # Calculate statistics
        durations = [split_times[i + 1] - split_times[i] for i in range(len(split_times) - 1)]
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        print(f"Segment Statistics:")
        print(f"Number of segments: {len(segments)}")
        print(f"Mean duration: {mean_duration:.20f} seconds")
        print(f"Duration std: {std_duration:.20f} seconds")

        # Generate debug plots
        if plot_segments > 0 and output_dir is not None:
            plot_segments = min(plot_segments, len(segments))
            fig, axes = plt.subplots(plot_segments, 1, figsize=(10, 4 * plot_segments))

            for i in range(plot_segments):
                ax = axes[i] if plot_segments > 1 else axes
                ax.plot(segments[i]['time'], segments[i]['signal'])
                ax.set_title(f'Segment {i}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.grid(True)

            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'segments_debug.png'))
            plt.close()

    return segments


def analyze_segments(segments: Dict[int, Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """
    Analyze time duration statistics of segments.

    Args:
        segments: Dictionary of segments

    Returns:
        Tuple of (mean_duration, std_duration)
    """
    durations = []
    for segment in segments.values():
        duration = segment['time'][-1] - segment['time'][0]
        durations.append(duration)

    return np.mean(durations), np.std(durations)


def drmos_trans_analysis(signal_data, debug=False):
    """
    Analyze DRMOS transient properties with error handling.
    Returns the most deviated dead time value and average rise time.

    Args:
        signal_data: Dictionary containing segmented signal data
        debug: Enable debug mode to print detailed information (default: False)

    Returns:
        Dict containing transient properties or default values if analysis fails
    """
    # Define default values for error cases or NaN results
    default_values = {
        'deadTime': 0.0,
        'riseTime': 0.0,
    }

    try:
        # Find indices where signal crosses specific thresholds
        def find_threshold_crossings(signal, thresholds):
            crossings = {}
            for threshold in thresholds:
                rising_cross_indices = np.where((signal[:-1] < threshold) & (signal[1:] >= threshold))[0]
                falling_cross_indices = np.where((signal[:-1] >= threshold) & (signal[1:] < threshold))[0]
                all_cross_indices = np.sort(np.concatenate([rising_cross_indices, falling_cross_indices]))
                crossings[threshold] = all_cross_indices
            return crossings

        def calculate_times(time, crossings):
            # Check if we have the expected number of crossings
            if (crossings.get(-0.05, []) is None or len(crossings.get(-0.05, [])) != 4 or
                    len(crossings.get(0.65, [])) != 2 or
                    len(crossings.get(5.8, [])) != 2):
                return None

            # Extract crossing times
            dead_times = time[crossings[-0.05]]
            low_voltage_times = time[crossings[0.65]]
            high_voltage_times = time[crossings[5.8]]

            # Calculate dead times
            dead_time1 = dead_times[1] - dead_times[0]
            dead_time2 = dead_times[3] - dead_times[2]
            dead_time = (dead_time1 + dead_time2) / 2

            # Calculate rise time
            rise_time = high_voltage_times[0] - low_voltage_times[0]

            return {
                'dead_time': dead_time,
                'rise_time': rise_time,
            }

        # Thresholds to check
        thresholds = [-0.05, 0.65, 5.8]

        # Store results for each signal
        results = {}
        all_dead_times = []
        all_rise_times = []

        # Process each signal
        for index, signal_dict in signal_data.items():
            time = signal_dict['time']
            signal = signal_dict['signal']

            # Find threshold crossings
            crossings = find_threshold_crossings(signal, thresholds)

            # Calculate times
            signal_times = calculate_times(time, crossings)

            if signal_times is None:
                logging.warning(f"Could not process signal {index} - incorrect threshold crossings")
                continue

            results[index] = signal_times

            # Collect times for overall statistics
            all_dead_times.append(signal_times['dead_time'])
            all_rise_times.append(signal_times['rise_time'])

        # Check if we have any valid measurements
        if not all_dead_times or not all_rise_times:
            logging.warning("No valid measurements found in signal data")
            return default_values

        # Calculate mean values
        dead_time_mean = np.mean(all_dead_times)
        rise_time_mean = np.mean(all_rise_times)

        # Find the most deviated dead time
        max_deviation = 0
        most_deviated_dead_time = all_dead_times[0]  # Default to first value

        for dead_time in all_dead_times:
            deviation = abs(dead_time - dead_time_mean)
            if deviation > max_deviation:
                max_deviation = deviation
                most_deviated_dead_time = dead_time

        # Check for NaN values
        if np.isnan(most_deviated_dead_time) or np.isnan(rise_time_mean):
            logging.warning("NaN values detected in calculations")
            return default_values

        trans_property = {
            'deadTime': most_deviated_dead_time,  # Return the most deviated dead time
            'riseTime': rise_time_mean,
        }

        # Debug mode output
        if debug:
            overall_stats = {
                'deadTime': {
                    'mean': dead_time_mean,
                    'most_deviated': most_deviated_dead_time,
                    'deviation': max_deviation,
                    'all_values': all_dead_times
                },
                'riseTime': {
                    'mean': rise_time_mean,
                    'std': np.std(all_rise_times)
                }
            }
            print("Individual Signal Results:")
            print(json.dumps(results, indent=10))
            print("\nOverall Statistics:")
            print(json.dumps(overall_stats, indent=10))

        return trans_property

    except Exception as e:
        logging.error(f"Error in DRMOS transient analysis: {str(e)}")
        return default_values


def findTransProperty(filename: str) -> Dict[str, Any]:
    """
    Extract transient properties from simulation results with error handling.
    If extraction fails, returns a dictionary with default values (0.0).

    Args:
        filename (str): Path to the simulation result file

    Returns:
        Dict[str, Any]: Dictionary containing transient properties or default values if extraction fails
    """
    # Define default values dict for error cases
    default_values = {
        'deadTime': 0.0,
        'riseTime': 0.0,
        'averageVoltage': 0.0
    }

    try:
        all_trace = extractTransTrace_psf(filename)
        # Extract time and SW_IDEAL signal, time is limited within 280us to 300us
        indices = [i for i, x in enumerate(all_trace['time']) if 280e-6 <= x <= 300e-6]

        # Check if we have enough data points
        if not indices:
            logging.warning(f"No data points found in specified time range for {filename}")
            return default_values

        extract_trace = {
            'time': [all_trace['time'][i] for i in indices],
            'signal': [all_trace['SW_IDEAL'][i] for i in indices]
        }

        output_voltage_values = [all_trace['OUT_IDEAL'][i] for i in indices]
        average_output_voltage = sum(output_voltage_values) / len(output_voltage_values)

        extract_trace_df = pd.DataFrame(extract_trace)
        extract_segments = split_signal(extract_trace_df)
        extract_drmos_trans_result = drmos_trans_analysis(extract_segments)

        extract_drmos_trans_result['averageVoltage'] = average_output_voltage

        return extract_drmos_trans_result

    except Exception as e:
        logging.error(f"Failed to extract transient properties from {filename}: {str(e)}")
        return default_values

# if __name__ == '__main__':
#     file = "/Users/hanwu/Downloads/tb_Efficiency/tb_Efficiency.raw/tran.tran.tran"
#     all_trace = extractTransTrace_psf(file)
#     # Extract time and SW_IDEAL signal, time is limited within 280us to 300us
#     indices = [i for i, x in enumerate(all_trace['time']) if 280e-6 <= x <= 300e-6]
#     extract_trace = {
#         'time': [all_trace['time'][i] for i in indices],
#         'signal': [all_trace['SW_IDEAL'][i] for i in indices]
#     }
#     extract_trace_df = pd.DataFrame(extract_trace)
#     print(f"Extracted trace is {extract_trace_df}")
#     extract_segments = split_signal(extract_trace_df)
#     print(f"Extracted segments are {extract_segments}")
#     extract_drmos_trans_result = drmos_trans_analysis(extract_segments)
#
#     output_voltage_values = [all_trace['OUT_IDEAL'][i] for i in indices]
#     average_output_voltage = sum(output_voltage_values) / len(output_voltage_values)
#
#     extract_drmos_trans_result['averageVoltage'] = average_output_voltage
#     print(f"Extracted DRMOSTrans result is {extract_drmos_trans_result}")

# if __name__ == '__main__':
#     # Load sample data
#     data = pd.read_csv('/Users/hanwu/Downloads/tb_Efficiency/tb_Efficiency.raw/demo.csv')
#     output_dir = '/Users/hanwu/Downloads/tb_Efficiency/tb_Efficiency.raw'
#     segments = split_signal(data, high_threshold=6.0, plot_segments=10, output_dir=output_dir, debug=True)
#     mean_duration, std_duration = analyze_segments(segments)
#     print(f"Mean duration: {mean_duration:.20f} seconds")
#     print(f"Duration std: {std_duration:.20f} seconds")
#     print(f"Output dict is {segments}")
#
#     result = drmos_trans_analysis(segments, debug=False)
#     print(f"Result is {result}")