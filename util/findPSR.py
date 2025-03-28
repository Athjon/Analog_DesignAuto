from util.extract_trace import extractACTrace
from util.util_func import find_closest_value_index, find_psr_turning_point
# from extract_trace import extractACTrace
# from util_func import find_closest_value_index, find_psr_turning_point
import math
from scipy import interpolate


def findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels):

    freq_name = 'freq'
    # vout_name = 'VOUT'

    psr_values = {}
    try:
        trace_dict = extractACTrace(file_path)

        freq_trace = trace_dict[freq_name]
        vout_trace = trace_dict[vout_name]

        for freq, label in zip(freq_list, freq_labels):
            index_freq = find_closest_value_index(freq_trace, freq)
            psr = -20 * math.log10(vout_trace[index_freq])

            if psr <= 0.0:
                psr = 0.0
                print(f"Warning: PSR at {label} is positive")

            psr_values[f"psr_{label}"] = psr

    except Exception as e:
        print("Warning: Extract PSR error", e)
        for label in freq_labels:
            psr_values[f"psr_{label}"] = 0.0

    return psr_values


def findPowerSupplyRejectionRatio_interpolated(vout_name, file_path, freq_list, freq_labels, find_spike=False):
    """
    Calculate Power Supply Rejection Ratio with interpolation and optional spike detection.

    Args:
        vout_name: Name of voltage output signal
        file_path: Path to simulation result file
        freq_list: List of frequencies to analyze
        freq_labels: Labels for each frequency
        find_spike: Whether to find PSR spike point

    Returns:
        Dict containing PSR values at specified frequencies and spike point if requested
    """
    freq_name = 'freq'
    psr_values = {}

    try:
        trace_dict = extractACTrace(file_path)
        freq_trace = trace_dict[freq_name]
        vout_trace = trace_dict[vout_name]

        # Calculate PSR trace and find spike if requested
        if find_spike:
            psr_trace = [-20 * math.log10(abs(v)) for v in vout_trace]
            spike_idx = find_psr_turning_point(psr_trace)

            if spike_idx != -1:
                spike_psr = psr_trace[spike_idx]
                if spike_psr < 0:
                    spike_psr = 0.0
                    print(f"Warning!!! PSR spike is negative: {spike_psr}, set to 0.0")
                psr_values["psr_spike"] = spike_psr
            else:
                psr_values["psr_spike"] = 0.0
                print("Warning: PSR spike not found")

        # Create interpolation function for regular PSR values
        f = interpolate.interp1d(freq_trace, vout_trace, kind='linear', fill_value='extrapolate')

        # Calculate PSR at specified frequencies
        for freq, label in zip(freq_list, freq_labels):
            vout_interpolated = f(freq)
            psr = -20 * math.log10(abs(vout_interpolated))

            if psr <= 0.0:
                psr = 0.0
                print(f"Warning: PSR at {label} is non-positive")

            psr_values[f"psr_{label}"] = psr

    except Exception as e:
        print("Warning: Extract PSR error", e)
        for label in freq_labels:
            psr_values[f"psr_{label}"] = 0.0
        if find_spike:
            psr_values["psr_spike"] = 0.0

    return psr_values


def findPowerSupplyRejectionRatio(file_path):

    vout_name = 'VOUT'

    freq_list = [100000, 1000000, 10000000]
    freq_labels = ['100k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result


def findPowerSupplyRejectionRatio_Yan(file_path):

    vout_name = 'VOUT'

    freq_list = [1000, 1000000, 100000000, 1000000000, 10000000000]
    freq_labels = ['1k', '1M', '100M', '1G', '10G']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result
# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio_Yan("/Users/hanwu/Downloads/tmp_20240508144031562245439/PSR.raw/ac.ac.encode")
# print(value_dict)


def findPowerSupplyRejectionRatio_Jianping(file_path):

    vout_name = 'VOUT'

    freq_list = [100, 1000, 10000, 100000, 1000000]
    freq_labels = ['100', '1k', '10k', '100k', '1M']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result
# Test the function with the provided file
# value_dict = findPowerSupplyRejectionRatio("/Users/hanwu/Downloads/ac.ac.encode")
# print(value_dict)

def findPowerSupplyRejectionRatio_Debashis(file_path):

    vout_name = 'VOUT'

    freq_list = [1000, 1000000, 10000000]
    freq_labels = ['1k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result


def findPowerSupplyRejectionRatio_Lab(file_path):

    vout_name = 'VOUT'

    freq_list = [1000, 1000000, 10000000]
    freq_labels = ['1k', '1M', '10M']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result

def findPowerSupplyRejectionRatio_Cai(file_path):

    vout_name = 'VOUT'

    freq_list = [1000000, 10000000, 100000000, 1000000000]
    freq_labels = ['1M', '10M', '100M', '1G']

    result = findPowerSupplyRejectionRatio_general(vout_name, file_path, freq_list, freq_labels)

    return result


def findPowerSupplyRejectionRatio_Haoqiang(file_path):

    vout_name = 'VOUT'

    freq_list = [10, 1000, 1000000, 10000000, 100000000]
    freq_labels = ['10', '1k', '1M', '10M', '100M']

    result = findPowerSupplyRejectionRatio_interpolated(vout_name, file_path, freq_list, freq_labels)

    return result

def findPowerSupplyRejectionRatio_AXS(file_path):

    vout_name = 'net3'

    freq_list = [100000]
    freq_labels = ['100k']

    result = findPowerSupplyRejectionRatio_interpolated(vout_name, file_path, freq_list, freq_labels, find_spike=True)

    return result

# if __name__ == "__main__":
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/PSR.raw/ac.ac"
#     value_dict = findPowerSupplyRejectionRatio_AXS(file_path)
#     print(value_dict)