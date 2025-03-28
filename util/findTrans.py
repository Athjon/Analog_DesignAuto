import os.path
# from extract_trace import extractTransTrace
# from extract_trace import extractTransTrace_psf
from util.extract_trace import extractTransTrace
from util.extract_trace import extractTransTrace_psf
import numpy as np
import bisect
import math
import matplotlib.pyplot as plt


def find_indices_in_range(nums, range_start, range_end):
    start_idx = bisect.bisect_left(nums, range_start)
    end_idx = bisect.bisect_right(nums, range_end)

    indices = list(range(start_idx, end_idx))

    return indices


def analyze_trans_file(file_path):
    """
    Parse the signals from the given file and return a dictionary with signal names as keys
    and their corresponding 2D numpy arrays (time and value) as values.
    """
    with open(file_path, "r") as file:
        content = file.readlines()

    # Parse the TRACE signals
    trace_signals = []
    parse_trace = False
    for line in content:
        if "TRACE" in line:
            parse_trace = True
            continue
        if parse_trace:
            if '"' in line:
                trace_signals.append(line.split('"')[1])
            else:
                break

    # Parse the VALUE section for time and signal values
    times = []
    values = []
    parse_value = False
    current_values = []
    for line in content:
        if "VALUE" in line:
            parse_value = True
            continue
        if "END" in line:
            if current_values:
                values.append(current_values)
                current_values = []
            parse_value = False
            continue
        if parse_value:
            if '"time"' in line:
                times.append(float(line.split()[-1]))
                if current_values:
                    values.append(current_values)
                    current_values = []
            elif '"' in line and not line.startswith('"time"'):
                current_values.append(float(line.split()[-1]))
            else:
                try:
                    current_values.append(float(line.strip()))
                except ValueError:
                    # Skip lines that cannot be converted to float
                    pass

    # Convert lists to numpy arrays
    times_array = np.array(times)
    values_array = np.array(values)

    # Create a dictionary with signal names as keys and their 2D numpy arrays as values
    signal_data = {}
    for idx, signal in enumerate(trace_signals[1:17]):
        combined_data = np.column_stack((times_array, values_array[:, idx]))
        signal_data[signal] = combined_data

    return signal_data


def findSlewRate(file_path):
    """
    Analyze the specified signal from the given file.

    Parameters:
    - file_path (str): Path to the file containing the signals.
    - signal_name (str): Name of the signal to be analyzed.

    Returns:
    - A plot of the specified signal against time.
    - A dictionary containing the average slew rates for rising and falling edges.
    """

    # Parse the signals
    signals = analyze_trans_file(file_path)
    # Define the output signal
    signal_name = "net3"

    # Check if the specified signal is present in the parsed data
    if signal_name not in signals:
        print(f"Signal {signal_name} not found in the file!")
        return

    # Extracting the specified signal and analyzing it
    signal_data = signals[signal_name]
    signal_time = signal_data[:, 0]
    signal_value = signal_data[:, 1]

    # Calculate the mid value
    mid_value = (np.max(signal_value) + np.min(signal_value)) / 2

    # Detect rising and falling edges
    rising_edges = []
    falling_edges = []

    for i in range(1, len(signal_value) - 1):
        if signal_value[i - 1] < mid_value < signal_value[i] and signal_value[i + 1] > mid_value:
            rising_edges.append(i)
        elif signal_value[i - 1] > mid_value > signal_value[i] and signal_value[i + 1] < mid_value:
            falling_edges.append(i)

    # Calculate the slew rate for each rising and falling edge
    slew_rates_up = []
    slew_rates_down = []

    for edge in rising_edges:
        start_value = mid_value + 0.1 * (np.max(signal_value) - mid_value)
        end_value = mid_value + 0.9 * (np.max(signal_value) - mid_value)
        start_indices = np.where(signal_value[edge:] < start_value)[0]
        end_indices = np.where(signal_value[edge:] > end_value)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            continue

        start_idx = start_indices[0] + edge
        end_idx = end_indices[0] + edge

        delta_v = signal_value[end_idx] - signal_value[start_idx]
        delta_t = signal_time[end_idx] - signal_time[start_idx]

        slew_rates_up.append(delta_v / delta_t)

    for edge in falling_edges:
        start_value = mid_value + 0.1 * (mid_value - np.min(signal_value))
        end_value = mid_value - 0.9 * (mid_value - np.min(signal_value))

        start_indices = np.where(signal_value[:edge] > start_value)[0]
        end_indices = np.where(signal_value[edge:] < end_value)[0]

        if len(start_indices) == 0 or len(end_indices) == 0:
            continue

        start_idx = start_indices[-1]
        end_idx = end_indices[0] + edge

        delta_v = signal_value[start_idx] - signal_value[end_idx]
        delta_t = signal_time[end_idx] - signal_time[start_idx]

        slew_rates_down.append(delta_v / delta_t)

    # Calculate the average slew rate and store in dictionaries
    slew_rate_up_val = np.mean(slew_rates_up).item() if slew_rates_up else None
    slew_rate_down_val = np.mean(slew_rates_down).item() if slew_rates_down else None

    # Plot the specified signal against time
    # plt.figure(figsize=(14, 6))
    # plt.plot(signal_time, signal_value, label=signal_name, color="blue")
    # plt.axhline(y=mid_value, color='r', linestyle='--', label="Mid value")
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    # plt.title(f"'{signal_name}' Signal Over Time")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # If slewRateUp or slewRateDown is None, return 0
    if slew_rate_up_val is None:
        slew_rate_up_val = 100.0
        print(f"Warning!!! slewRateUp is not found, set to 100.0")
    if slew_rate_down_val is None:
        slew_rate_down_val = 100.0
        print(f"Warning!!! slewRateDown is not found, set to 100.0")
    if slew_rate_up_val < 0.0:
        slew_rate_up_val = 100.0
        print(f"Warning!!! slewRateUp is negative, set to 100.0")
    if slew_rate_down_val < 0.0:
        slew_rate_down_val = 100.0
        print(f"Warning!!! slewRateDown is negative, set to 100.0")

    return {"slewRateUp": slew_rate_up_val, "slewRateDown": slew_rate_down_val}


# dict = findSlewRate("/Users/hanwu/ML/AnalogDesignAuto/resultParse/spectreEnv/spectreTmpFile/
# tmp_202308181553507320/Trans.raw/tran.tran.tran.encode")
# print(dict)


def findShoot_general(filename, time_ranges, stable_voltage, output_voltage_label="VOUT", file_size_threshold=500):
    """
    Extract the overshoot and undershoot value from trans file

    Args:
    - filename: Path to the file to be processed.
    - time_ranges: A list of tuples, each defining a time range.
    - stable_voltage: The stable voltage value.

    Returns:
    - Overshoot and undershoot value
    """

    trans_dict = extractTransTrace(filename)
    time_series = trans_dict["time"]
    # print(f"Debug!!! time_series: {time_series} \n with length: {len(time_series)}")
    vout_trace = trans_dict[output_voltage_label]
    # print(f"Debug!!! vout_trace: {vout_trace} \n with length: {len(vout_trace)}")

    # Clip time according to time_ranges
    time_undershoot_index = find_indices_in_range(time_series, time_ranges[0][0], time_ranges[0][1])
    # print(f"Debug!!! time_undershoot_index: {time_undershoot_index}")
    time_overshoot_index = find_indices_in_range(time_series, time_ranges[1][0], time_ranges[1][1])
    # print(f"Debug!!! time_overshoot_index: {time_overshoot_index}")
    vout_undershoot = [vout_trace[i] for i in time_undershoot_index]
    vout_overshoot = [vout_trace[i] for i in time_overshoot_index]
    # print(f"Debug!!! vout_undershoot: {vout_undershoot}")
    # print(f"Debug!!! vout_overshoot: {vout_overshoot}")

    # Calculate overshoot and undershoot,
    # undershoot: Vout@50us -Vout_clip1_min, overshoot: Vout_clip2_max - Vout@100us
    vout_undershoot_min = np.min(vout_undershoot)
    vout_overshoot_max = np.max(vout_overshoot)
    # Find the neset value to 50us and 100us
    vout_undershoot_base = vout_undershoot[0]
    vout_overshoot_base = vout_overshoot[0]
    # print(f"Debug, vout_undershoot_base: {vout_undershoot_base} and vout_overshoot_base: {vout_overshoot_base}")
    # print(f"Debug, vout_undershoot_min: {vout_undershoot_min} and vout_overshoot_max: {vout_overshoot_max}")

    undershoot = vout_undershoot_base - vout_undershoot_min
    overshoot = vout_overshoot_max - vout_overshoot_base

    # Determine whether the stable voltage is regulated to 1.2V (pre-defined)
    # Find the mid-value in vout_undershoot
    stable_high_load_voltage = vout_undershoot[-1]
    stable_light_load_voltage = vout_overshoot[-1]
    # print(f"Debug, stable_light_load_voltage: {stable_light_load_voltage}")
    # print(f"Debug, stable_high_load_voltage: {stable_high_load_voltage}")

    # Calculate penalty coefficient for punishing the max difference between ideal and real stable voltage
    # max_stable_voltage_diff = max(abs(stable_high_load_voltage - stable_voltage),
    #                               abs(stable_light_load_voltage - stable_voltage))
    # print(f"Debug, max_stable_voltage_diff: {max_stable_voltage_diff}")
    # penalty_coeff = max_stable_voltage_diff/stable_voltage
    # print(f"Debug, penalty_coeff: {penalty_coeff}")

    if stable_high_load_voltage >= stable_voltage * 1.04 or stable_high_load_voltage <= stable_voltage * 0.96:
        print("Warning! This LDO cannot be regulated to VREF under high load.")
        overshoot = 100.0
        undershoot = 100.0
    if stable_light_load_voltage >= stable_voltage * 1.04 or stable_light_load_voltage <= stable_voltage * 0.96:
        print("Warning! This LDO cannot be regulated to VREF under light load.")
        overshoot = 100.0
        undershoot = 100.0
    if stable_high_load_voltage == 0.0 or stable_light_load_voltage == 0.0:
        print(f"Warning! No shoot be found.")
        overshoot = 100.0
        undershoot = 100.0
    if overshoot == 0.0 or undershoot == 0.0:
        print(f"Warning! Shoot is zero. Too good to be true")
        overshoot = 100.0
        undershoot = 100.0
    if os.path.getsize(filename) > file_size_threshold * 1024:
        print(f"Warning! {filename} is larger than {file_size_threshold}K, the system is highly like to be unstable")
        overshoot = 100.0
        undershoot = 100.0
    # if overshoot >= stable_voltage or undershoot >= stable_voltage:
    #     print("Warning! Shoot is too large.")
    #     overshoot = 100.0
    #     undershoot = 100.0
    else:
        pass

    # print(f"Debug, overshoot: {overshoot} and undershoot: {undershoot}")

    # overshoot = overshoot * (1 + penalty_coeff)
    # undershoot = undershoot * (1 + penalty_coeff)

    return {"overShoot": overshoot, "underShoot": undershoot}


def findShoot(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.0)
    return result


# Test Code
# file = "/Users/hanwu/Downloads/Log_N65/Mohamed/select_point/tmp_20240521235143711221896/Trans.raw/tran.tran.tran.encode.encode"
# print(findShoot(file))


def findShoot_Jianping(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 0.5, "VOUT", 75)
    return result


# Test Code
# file = "/Users/hanwu/Downloads/Log_N65/Jianping/select_point/tmp_20240518030545162069028/Trans_1_2V.raw/tran.tran.tran.encode"
# print(findShoot_Jiangping(file))

def findShoot_Debashis(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.6)
    return result

# Test Code
# file = "/Users/hanwu/Downloads/Log_N65/Debashis/select_point/tmp_20240520224716498574920/Trans_9m.raw/tran.tran.tran.encode"
# print(findShoot_Debashis(file))

def findShoot_Line_Reg_Debashis(filename):
    result = findShoot_general(filename, [(7.5e-6, 12.5e-6), (2.5e-6, 7.5e-6)], 1.6)
    return result


def findShoot_Line_Reg_Jianping(filename):
    result = findShoot_general(filename, [(0.00023, 0.00046), (0.00004, 0.00023)], 0.5)
    return result


def findShoot_Jianping_V(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.0)
    return result


def findShoot_KNL(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.0)
    return result


def findShoot_Lab(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.2)
    return result


def findShoot_Cai(filename):
    result = findShoot_general(filename, [(2.5e-6, 7.5e-6), (7.5e-6, 12.5e-6)], 1.0)
    return result


def findShoot_Haoqiang(filename):
    output_voltage_label = "VOUT_OS"
    file_size_threshold = 150
    result = findShoot_general(filename, [(1.0e-6, 4.5e-6), (4.5e-6, 9.0e-6)], 1.0, output_voltage_label, file_size_threshold)
    return result

# Test Code
# file = "/Users/hanwu/Downloads/Test_Run/Trans.raw/tran.tran.tran.encode"
# print(findShoot_Haoqiang(file))


def findEff_general(filename, time_ranges, output_voltage_label, load_current, input_voltage, input_current_label):
    """
    Extract the efficiency value

    Args:
    - filename: Path to the file to be processed.
    - time_ranges: A list of tuples, each defining a time range.
    - output_voltage_label: The output voltage label.
    - load_current: The load current value.
    - input_voltage: The input voltage value.
    - input_voltage_label: The input voltage label.

    Returns:
    - Efficiency
    """

    def calculate_variance_trapz(x, y, mean):
        squared_diff = (y - mean)**2
        variance = np.trapz(squared_diff, x) / (x[-1] - x[0])
        return variance

    try:
        trans_dict = extractTransTrace(filename)
        time_series = trans_dict["time"]
        # print(f"Debug!!! time_series with length: {len(time_series)}")
        output_voltage_series = trans_dict[output_voltage_label]
        # print(f"Debug!!! output_voltage_series with length: {len(output_voltage_series)}")
        input_current_series = trans_dict[input_current_label]
        # print(f"Debug!!! input_current_series with length: {len(input_current_series)}")

        # Clip time according to time_ranges
        time_index = find_indices_in_range(time_series, time_ranges[0], time_ranges[1])
        extracted_time_series = [time_series[i] for i in time_index]
        # print(f"Debug!!! time_index with length: {len(time_index)}")
        extracted_output_voltage_series = [output_voltage_series[i] for i in time_index]
        # print(f"Debug!!! extracted_output_voltage_series with length: {len(extracted_output_voltage_series)}")
        extracted_input_current_series = [input_current_series[i] for i in time_index]
        # print(f"Debug!!! extracted_input_current_series with length: {len(extracted_input_current_series)}")

        # Plot extracted_output_voltage_series and extracted_input_current_series against time, keep plt open
        # plt.figure(figsize=(14, 6))
        # plt.plot(extracted_time_series, extracted_output_voltage_series, label=output_voltage_label, color="blue")
        # plt.scatter(extracted_time_series, extracted_input_current_series, label=input_current_label, color="red")
        # plt.plot(extracted_time_series, extracted_input_current_series, label=input_current_label, color="red")
        # plt.xlabel("Time")
        # plt.ylabel("Value")
        # plt.title(f"'{output_voltage_label}' and '{input_current_label}' Signal Over Time")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # avg_output_voltage is given by the integral of extracted_output_voltage_series in the range of
        # extracted_time_series and then divided by the time range
        avg_output_voltage = (np.trapz(extracted_output_voltage_series, x=extracted_time_series) /
                              (time_ranges[1] - time_ranges[0]))
        output_voltage_variance = calculate_variance_trapz(extracted_time_series, extracted_output_voltage_series, avg_output_voltage)
        # avg_input_current is given by the integral of extracted_input_current_series in the range of
        # extracted_time_series and then divided by the time range
        avg_input_current = (np.trapz(extracted_input_current_series, x=extracted_time_series) /
                             (time_ranges[1] - time_ranges[0]))
        # print(f"Debug, avg_output_voltage: {avg_output_voltage}")
        # print(f"Debug, avg_input_current: {avg_input_current}")
        avg_input_current = abs(avg_input_current)

        # Calculate efficiency
        efficiency = avg_output_voltage * load_current / (input_voltage * avg_input_current)
    except Exception as e:
        print(f"Error!!! {e}")
        efficiency = 0.0
        avg_output_voltage = 0.0
        output_voltage_variance = 100.0

    if avg_output_voltage < 0.0:
        avg_output_voltage = 0.0
        print(f"Warning!!! avg_output_voltage is negative, set to 0.0")

    if efficiency < 0 or efficiency > 1:
        efficiency = 0.0
        print(f"Warning!!! efficiency is {efficiency} and out of range, set to 0.0")

    # If any value is NaN or Inf, set to 0
    for value in [efficiency, avg_output_voltage, output_voltage_variance]:
        if math.isnan(value) or math.isinf(value):
            value = 0.0
            print(f"Warning!!! value is {value} and is NaN or Inf, set to 0.0")

    return {"efficiency": efficiency, "vo_mean": avg_output_voltage,
            "vo_var": output_voltage_variance}


def findEff_Buck(filename):
    result = findEff_general(filename, (5e-5, 6e-5), "VOUT", 1, 3.3, "V0:p")
    return result

# Test Code

# filename = "/Users/hanwu/Downloads/tran.tran.tran.encode"
# print(findEff_Buck(filename))

def findEff_DRMOS(filename, debug=False):
    """
    Calculate efficiency for DRMOS design. Extract time series data, calculate input and output power,
    and return the efficiency. Power calculation is based on time-domain average of current and voltage.

    Args:
        filename (str): Path to the processed spectre simulation result file
        debug (bool): If True, plot debug info and print intermediate results

    Returns:
        dict: Dictionary containing input power, output power and efficiency
    """
    try:
        import numpy as np

        # Extract time series data
        trans_dict = extractTransTrace_psf(filename)
        time_series = trans_dict["time"]

        # Check if required signals exist
        required_signals = ['VS_VIN:p', 'VIN', 'OUT_IDEAL', 'IS_LOAD:sink']
        for signal in required_signals:
            if signal not in trans_dict:
                print(f"Warning!!! {signal} not found in trace data")
                return {"inputPower": 0.0, "outputPower": 0.0, "Efficiency": 0.0}

        # Extract signals
        vs_vin_current = trans_dict['VS_VIN:p']
        vin_voltage = trans_dict['VIN']
        out_voltage = trans_dict['OUT_IDEAL']
        load_current = trans_dict['IS_LOAD:sink']

        # Get data in specified time range (250us - 300us)
        time_indices = find_indices_in_range(time_series, 250e-6, 300e-6)
        extracted_time = [time_series[i] for i in time_indices]
        extracted_vs_vin = [vs_vin_current[i] for i in time_indices]
        extracted_vin = [vin_voltage[i] for i in time_indices]
        extracted_out = [out_voltage[i] for i in time_indices]
        extracted_load = [load_current[i] for i in time_indices]

        # Calculate time-domain averages using numpy's trapz
        time_period = extracted_time[-1] - extracted_time[0]
        avg_vs_vin = abs(np.trapz(extracted_vs_vin, extracted_time) / time_period)
        avg_vin = abs(np.trapz(extracted_vin, extracted_time) / time_period)
        avg_out = abs(np.trapz(extracted_out, extracted_time) / time_period)
        avg_load = abs(np.trapz(extracted_load, extracted_time) / time_period)

        # Calculate power using averaged values
        input_power = avg_vs_vin * avg_vin
        output_power = avg_out * avg_load

        # Calculate efficiency
        efficiency = output_power / input_power if input_power != 0 else 0.0

        # Debug plotting and printing
        if debug:
            import matplotlib.pyplot as plt

            # Create figure for each sequence
            sequences = [
                (extracted_vs_vin, 'Input Current (VS_VIN:p)', 'Current (A)'),
                (extracted_vin, 'Input Voltage (VIN)', 'Voltage (V)'),
                (extracted_out, 'Output Voltage (OUT_IDEAL)', 'Voltage (V)'),
                (extracted_load, 'Load Current (IS_LOAD:sink)', 'Current (A)')
            ]

            for idx, (sequence, title, ylabel) in enumerate(sequences, 1):
                plt.figure(figsize=(10, 6))
                plt.plot(extracted_time, sequence, 'b-', linewidth=1.5)
                plt.title(title)
                plt.xlabel('Time (s)')
                plt.ylabel(ylabel)
                plt.grid(True)
                plt.tight_layout()

            plt.show()

            print("\nTime Domain Analysis Information:")
            print(f"Time range: {extracted_time[0] * 1e6:.2f}us to {extracted_time[-1] * 1e6:.2f}us")
            print(f"Total time period: {time_period * 1e6:.2f}us")
            print(f"Number of sample points: {len(extracted_time)}")
            print(f"Average time step: {np.mean(np.diff(extracted_time)) * 1e9:.2f}ns")

            print("\nVoltage Analysis:")
            print(f"Input voltage - Range: {min(extracted_vin):.6f}V to {max(extracted_vin):.6f}V")
            print(f"Input voltage - Time domain average: {avg_vin:.6f}V")
            print(f"Output voltage - Range: {min(extracted_out):.6f}V to {max(extracted_out):.6f}V")
            print(f"Output voltage - Time domain average: {avg_out:.6f}V")

            print("\nCurrent Analysis:")
            print(f"Input current - Range: {min(extracted_vs_vin):.6f}A to {max(extracted_vs_vin):.6f}A")
            print(f"Input current - Time domain average: {avg_vs_vin:.6f}A")
            print(f"Load current - Range: {min(extracted_load):.6f}A to {max(extracted_load):.6f}A")
            print(f"Load current - Time domain average: {avg_load:.6f}A")

            print("\nPower Analysis:")
            print(f"Input power (based on time-domain averages): {input_power:.6f}W")
            print(f"Output power (based on time-domain averages): {output_power:.6f}W")
            print(f"Overall efficiency: {efficiency * 100:.2f}%")

            print(f"Average input voltage: {avg_vin:.6f}V")
            print(f"Average input current: {avg_vs_vin:.6f}A")
            print(f"Average output voltage: {avg_out:.6f}V")
            print(f"Average output current: {avg_load:.6f}A")

            # Save VS_VIN:p with time series to csv file
            import csv
            with open('VS_VIN_p.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time (s)', 'Current (A)'])
                for i in range(len(extracted_time)):
                    writer.writerow([extracted_time[i], extracted_vs_vin[i]])
            # Save OUT_IDEAL with time series to csv file
            with open('OUT_IDEAL.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Time (s)', 'Voltage (V)'])
                for i in range(len(extracted_time)):
                    writer.writerow([extracted_time[i], extracted_out[i]])


        return {
            "inputPower": input_power,
            "outputPower": output_power,
            "efficiency": efficiency
        }

    except Exception as e:
        print(f"Error in findEff_DRMOS: {str(e)}")
        return {"inputPower": 0.0, "outputPower": 0.0, "efficiency": 0.0}

# Test Code
# demo_file = "/Users/hanwu/Downloads/Compare_Netlist/tb_Efficiency/tb_Efficiency.raw/tran.tran.tran"
# print(findEff_DRMOS(demo_file, debug=True))

def findShoot_AXS(filename):

    try:
        trace_dict = extractTransTrace_psf(filename)
        vout_name = 'net2'
        vout_trace = trace_dict[vout_name]
        time_series = trace_dict["time"]
        # print(f"vout_trace is {vout_trace} \n with length {len(vout_trace)} \n time_series is {time_series} "
        #       f"\n with length {len(time_series)}")

        # Extract net2 trace data within 0-75us
        start_up_ranges = [0, 75e-6]
        start_up_indices = find_indices_in_range(time_series, start_up_ranges[0], start_up_ranges[1])
        # print(f"Debug!!! start_up_indices is {start_up_indices}")
        vout_start_up_trace = [vout_trace[i] for i in start_up_indices]
        vout_startup_stable = vout_start_up_trace[-1]
        vout_startup_max = max(vout_start_up_trace)
        # print(f"Debug!!! vout_startup_stable is {vout_startup_stable} and vout_startup_max is {vout_startup_max}")

        startup_shoot = vout_startup_max - vout_startup_stable

        # Extract net2 trace data within 0.1-1.0ms
        time_range = [0.1e-3, 1.0e-3]
        time_indices = find_indices_in_range(time_series, time_range[0], time_range[1])
        vout_trace_selected = [vout_trace[i] for i in time_indices]
        vout_max = max(vout_trace_selected)
        vout_min = min(vout_trace_selected)
        overshoot = vout_max - vout_startup_stable
        undershoot = vout_startup_stable - vout_min

        if vout_startup_stable <= 1.1 or vout_startup_stable >= 1.3:
            print(f"Warning!!! vout_startup_stable is {vout_startup_stable}, set to 100.0")
            startup_shoot = 100.0

        if overshoot < 0 or undershoot < 0:
            print(f"Warning!!! overshoot is {overshoot} and undershoot is {undershoot}, set to 100.0")
            overshoot = 100.0
            undershoot = 100.0

    except Exception as e:
        print(f"Error in findShoot_AXS: {str(e)}")
        startup_shoot = 100.0
        overshoot = 100.0
        undershoot = 100.0

    return {"startupShoot": startup_shoot, "overShoot": overshoot, "underShoot": undershoot}

# if __name__ == "__main__":
#     demo_file = "/Users/hanwu/Downloads/Netlist_AXS/Load_Reg_Trans.raw/tran.tran.tran"
#     print(findShoot_AXS(demo_file))
