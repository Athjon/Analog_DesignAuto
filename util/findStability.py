import re
import os
import subprocess
import numpy as np
# from extract_bode import analyze_loop_gain_only_value, extract_dc_gain
from util.extract_bode import analyze_loop_gain_only_value, extract_dc_gain


def findLoopGain(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

        for i in range(len(lines)):
            if "VALUE" in lines[i]:
                # Extract the value from the second line below the "VALUE" line
                value_line = lines[i + 2]
                if '"loopGain"' in value_line:
                    # Split the line using spaces and then extract the value inside the parentheses
                    value_str = value_line.split()[1].lstrip("(")
                    value_float = float(value_str)
                else:
                    value_float = -100.0
                    print("Warning: loopGain not Existing, set to -100.0")
                    return {"loopGain": value_float}


def findPhaseMarginAndGBW(filename):
    with open(filename, 'r') as file:
        file_content = file.read()

    header_type_content = re.search(r'HEADER(.*?)TYPE', file_content, re.DOTALL)
    if header_type_content:
        header_type_content = header_type_content.group(1)
    else:
        print("Warning from findPhaseMarginAndGBW: HEADER or TYPE section not found in the file.")
        return {"phaseMargin": 0.0, "gainBandWidth": 0.0}

    required_keywords = ["phaseMargin", "phaseMarginFrequency"]
    has_keywords = all(keyword in header_type_content for keyword in required_keywords)

    phase_margin = phase_margin_frequency = 0.0

    if has_keywords:
        phase_margin_match = re.search(r'"phaseMargin"\s+"([\d.+e]+)\s+Deg"', header_type_content)
        phase_margin_frequency_match = re.search(r'"phaseMarginFrequency"\s+"([\d.+e]+)\s+Hz"', header_type_content)

        if phase_margin_match and phase_margin_frequency_match:
            phase_margin = float(phase_margin_match.group(1))
            phase_margin_frequency = float(phase_margin_frequency_match.group(1))
        else:
            print("Warning: phaseMargin or phaseMarginFrequency not properly formatted, set to 0.0")
    else:
        print("Warning: phaseMargin or phaseMarginFrequency not existing, set to 0.0")

    gain_bandwidth = phase_margin_frequency

    return {"phaseMargin": phase_margin, "gainBandWidth": gain_bandwidth}


def findPhaseMarginAndGBW_Jianping(encoded_file_path):
    """
    Custom function to analyze phase margin and gain bandwidth with specific checks.
    Added check for negative phase values before 1MHz.

    Parameters:
    encoded_file_path: Absolute path to the stb.margin.stb.encode file

    Returns:
    tuple: (phase_margin, gain_bandwidth)
        Returns the same values as findPhaseMarginAndGBW if phase checks pass
        Both values will be 0 if phase requirements are not met
    """
    try:
        # Extract directory path from the encoded file path
        directory_path = os.path.dirname(encoded_file_path)

        # Construct path for stb.stb file in the same directory
        file_to_process = os.path.join(directory_path, "stb.stb")
        if not os.path.exists(file_to_process):
            print(f"Warning!!! stb.stb not found in {directory_path}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Generate processed file path
        processed_file = os.path.join(directory_path, "stb.stb.encode")

        # Run psf command
        try:
            subprocess.run(f"psf {file_to_process} -o {processed_file}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning!!! Error running psf command: {e}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Check if processed file was created
        if not os.path.exists(processed_file):
            print(f"Warning!!! Processed file not created at {processed_file}")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Analyze loop gain using extract_bode function
        df = analyze_loop_gain_only_value(processed_file)

        # Check for negative phase values before 1MHz
        freq_mask = df['Frequency (Hz)'] <= 1e6
        phase_values = df[freq_mask]['Phase (degrees)']
        if (phase_values < 0).any():
            print("Warning!!! Negative phase value detected before 1MHz")
            return {"phaseMargin": 0, "gainBandWidth": 0}

        # Check phase requirements at specific frequencies
        test_freqs = [1e4, 1e5, 1e6]
        for test_freq in test_freqs:
            # Find the closest frequency in the data
            freq_array = df['Frequency (Hz)'].values
            closest_idx = np.argmin(np.abs(freq_array - test_freq))
            actual_freq = freq_array[closest_idx]
            phase = df.iloc[closest_idx]['Phase (degrees)']

            print(f"Warning!!! Checking phase at {actual_freq:.2e} Hz (closest to {test_freq:.2e} Hz): {phase:.2f} degrees")

            if phase > 120:
                print(f"Warning!!! Phase requirement not met: {phase:.2f} degrees > 120 degrees at {actual_freq:.2e} Hz")
                return {"phaseMargin": 0, "gainBandWidth": 0}

        # If all phase checks pass, use original findPhaseMarginAndGBW function
        # print("\nAll phase requirements met!")
        return findPhaseMarginAndGBW(encoded_file_path)

    except Exception as e:
        print(f"Warning!!! Error in findPhaseMarginAndGBW_Jianping: {e}")
        return {"phaseMargin": 0, "gainBandWidth": 0}


def findStability_AXS(filename):
    with open(filename, 'r') as file:
        file_content = file.read()

    header_type_content = re.search(r'HEADER(.*?)TYPE', file_content, re.DOTALL)
    if header_type_content:
        header_type_content = header_type_content.group(1)
    else:
        print("Warning from findPhaseMarginAndGBW: HEADER or TYPE section not found in the file.")
        return {"dcGain": 0.0, "gainMargin": 0.0, "phaseMargin": 0.0, "gainBandWidth": 0.0}

    required_keywords = ["gainMargin", "phaseMargin", "phaseMarginFrequency"]
    has_keywords = all(keyword in header_type_content for keyword in required_keywords)

    gain_margin = phase_margin = phase_margin_frequency = 0.0

    if has_keywords:
        gain_margin_match = re.search(r'"gainMargin"\s+"([\d.+e]+)\s+dB"', header_type_content)
        phase_margin_match = re.search(r'"phaseMargin"\s+"([\d.+e]+)\s+Deg"', header_type_content)
        phase_margin_frequency_match = re.search(r'"phaseMarginFrequency"\s+"([\d.+e]+)\s+Hz"', header_type_content)

        if phase_margin_match and phase_margin_frequency_match:
            try:
                gain_margin = float(gain_margin_match.group(1))
                phase_margin = float(phase_margin_match.group(1))
                phase_margin_frequency = float(phase_margin_frequency_match.group(1))

                gain_margin = max(0.0, gain_margin)
                phase_margin = max(0.0, phase_margin)
                phase_margin_frequency = max(0.0, phase_margin_frequency)
            except (ValueError, TypeError, AttributeError):
                print("Warning: Error converting values to float, resetting to 0.0")
                gain_margin = phase_margin = phase_margin_frequency = 0.0
        else:
            print("Warning: gainMargin or phaseMargin or phaseMarginFrequency not properly formatted, set to 0.0")
    else:
        print("Warning: gainMargin or phaseMargin or phaseMarginFrequency not existing, set to 0.0")

    gain_bandwidth = phase_margin_frequency

    # Change filename to stb.stb under same directory
    stb_file = filename.replace("stb.margin.stb", "stb.stb")
    try:
        dc_gain = extract_dc_gain(stb_file)
        dc_gain = max(0.0, dc_gain)
    except Exception as e:
        print(f"Warning: Error extracting DC gain: {str(e)}, setting to 0.0")
        dc_gain = 0.0

    return {
        "gain": dc_gain,
        "marginGain": gain_margin,
        "phaseMargin": phase_margin,
        "gainBandWidth": gain_bandwidth
    }

# if __name__ == "__main__":
#     phase_margin_gbw_file = "/Users/hanwu/Downloads/Netlist_AXS/Stability.raw/stb.margin.stb"
#     phase_margin_gbw = findStability_AXS(phase_margin_gbw_file)
#     print(f"Phase Margin and Gain Bandwidth: {phase_margin_gbw}")
