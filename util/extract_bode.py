import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_psf_file(file_path):
    """
    Read stability analysis data from PSF format file

    Parameters:
    file_path: Path to the PSF file

    Returns:
    tuple: (frequency array, loop gain array in complex form)
    """
    freq_list = []
    loopgain_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Flag for VALUE section
        in_value_section = False
        current_freq = None

        for line in lines:
            line = line.strip()

            # Check if entering VALUE section
            if line == "VALUE":
                in_value_section = True
                continue

            if line == "END":
                break

            if in_value_section:
                if '"freq"' in line:
                    current_freq = float(line.split()[-1])
                elif '"loopGain"' in line:
                    if current_freq is not None:
                        # Parse complex form (real imag)
                        complex_str = line.split('"loopGain"')[-1].strip()
                        if complex_str.startswith('(') and complex_str.endswith(')'):
                            complex_str = complex_str[1:-1]  # Remove parentheses
                            real, imag = map(float, complex_str.split())
                            freq_list.append(current_freq)
                            loopgain_list.append(complex(real, imag))
                            current_freq = None

    return np.array(freq_list), np.array(loopgain_list)


def extract_loop_gain_data(file_path):
    """
    Extract frequency and loop gain data from PSF file and calculate magnitude(dB) and phase(degrees)

    Parameters:
    file_path: Path to the PSF file

    Returns:
    pandas.DataFrame: DataFrame containing frequency, magnitude, phase, and complex components
    """
    # Read PSF file
    freq, complex_gains = read_psf_file(file_path)

    # Extract real and imaginary parts
    real = complex_gains.real
    imag = complex_gains.imag

    # Calculate magnitude(dB) and phase(degrees)
    magnitude = 20 * np.log10(np.sqrt(real ** 2 + imag ** 2))  # Convert to dB
    phase = np.degrees(np.arctan2(imag, real))  # Convert to degrees

    # Create DataFrame
    df = pd.DataFrame({
        'Frequency (Hz)': freq,
        'Magnitude (dB)': magnitude,
        'Phase (degrees)': phase,
        'Real': real,
        'Imaginary': imag
    })

    return df


def plot_bode(df, title="Bode Plot", show_grid=True, figure_size=(12, 8)):
    """
    Generate Bode plot with magnitude and phase responses

    Parameters:
    df: DataFrame containing frequency response data
    title: Plot title
    show_grid: Boolean to show/hide grid
    figure_size: Tuple defining figure dimensions

    Returns:
    tuple: (figure, axes)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size)
    fig.suptitle(title)

    # Frequency data
    freq = df['Frequency (Hz)']

    # Magnitude plot
    ax1.semilogx(freq, df['Magnitude (dB)'], 'b-', linewidth=2)
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.grid(show_grid, which="both")
    ax1.set_title('Magnitude Response')

    # Add 0dB reference line
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Phase plot
    ax2.semilogx(freq, df['Phase (degrees)'], 'b-', linewidth=2)
    ax2.set_ylabel('Phase (degrees)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(show_grid, which="both")
    ax2.set_title('Phase Response')

    # Add -180° reference line
    ax2.axhline(y=-180, color='r', linestyle='--', alpha=0.5)

    # Adjust layout
    plt.tight_layout()

    return fig, (ax1, ax2)


def analyze_stability(df):
    """
    Analyze system stability metrics

    Parameters:
    df: DataFrame containing frequency response data

    Returns:
    dict: Stability metrics including phase margin, gain margin, and crossover frequencies
    """
    # Find gain and phase crossover points
    gain_crossover_idx = (df['Magnitude (dB)'].abs()).idxmin()  # Point closest to 0dB
    phase_margin = 180 + df.iloc[gain_crossover_idx]['Phase (degrees)']
    gain_crossover_freq = df.iloc[gain_crossover_idx]['Frequency (Hz)']

    # Find phase crossover point (closest to -180°)
    phase_crossover_idx = (df['Phase (degrees)'] + 180).abs().idxmin()
    gain_margin = -df.iloc[phase_crossover_idx]['Magnitude (dB)']
    phase_crossover_freq = df.iloc[phase_crossover_idx]['Frequency (Hz)']

    return {
        'Phase Margin': phase_margin,
        'Gain Margin': gain_margin,
        'Gain Crossover Frequency': gain_crossover_freq,
        'Phase Crossover Frequency': phase_crossover_freq
    }


def analyze_loop_gain(file_path):
    """
    Main function: Analyze loop gain and generate report

    Parameters:
    file_path: Path to the PSF file

    Returns:
    tuple: (DataFrame with frequency response data, stability metrics, figure, axes)
    """
    # Extract data
    df = extract_loop_gain_data(file_path)

    # Stability analysis
    stability = analyze_stability(df)

    # Generate Bode plot
    fig, axes = plot_bode(df)

    # Annotate key points
    gain_crossover_freq = stability['Gain Crossover Frequency']
    phase_crossover_freq = stability['Phase Crossover Frequency']

    # Annotate gain margin
    axes[0].annotate(f'Gain Margin: {stability["Gain Margin"]:.1f} dB',
                     xy=(phase_crossover_freq, 0),
                     xytext=(phase_crossover_freq, 10),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # Annotate phase margin
    axes[1].annotate(f'Phase Margin: {stability["Phase Margin"]:.1f}°',
                     xy=(gain_crossover_freq, -180),
                     xytext=(gain_crossover_freq, -150),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    return df, stability, fig, axes

def analyze_loop_gain_only_value(file_path):
    """
    Main function: Analyze loop gain and generate report

    Parameters:
    file_path: Path to the PSF file

    """
    # Extract data
    df = extract_loop_gain_data(file_path)

    return df

def extract_dc_gain(file_path):

    try:
        df = analyze_loop_gain_only_value(file_path)
        # Extract first line of the Magnitude value
        dc_gain = df.iloc[0]['Magnitude (dB)']
    except Exception as e:
        print(f"Warning: {e}. Setting DC Gain to default (0.0).")
        dc_gain = 0.0

    return dc_gain

# if __name__ == "__main__":
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/Stability.raw/stb.stb"
    # df, stability, fig, axes = analyze_loop_gain(file_path)
    # print(df)
    #
    # # Print stability analysis results
    # print("\nStability Analysis Results:")
    # print(f"Phase Margin: {stability['Phase Margin']:.2f} degrees")
    # print(f"Gain Margin: {stability['Gain Margin']:.2f} dB")
    # print(f"Gain Crossover Frequency: {stability['Gain Crossover Frequency']:.2e} Hz")
    # print(f"Phase Crossover Frequency: {stability['Phase Crossover Frequency']:.2e} Hz")

    # dc_gain = extract_dc_gain(file_path)
    # print(f"DC Gain: {dc_gain:.2f} dB")
    #
    # plt.show()