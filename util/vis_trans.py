from matplotlib import pyplot as plt
import numpy as np
import os

from extract_trace import extractTransTrace


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'Trans.raw', 'tran.tran.tran.encode')
        if os.path.isfile(target_file):
            trace_data = extractTransTrace(target_file)
            results[subdir] = {
                'time': trace_data['time'],
                'VOUT': trace_data['VOUT']
            }
            print(trace_data['time'])
            print(trace_data['VOUT'])
    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.plot(data['time'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['time'], data['VOUT'], label=subdir, color='blue', linewidth=1)
            plt.plot(data['time'], data['VOUT'], label=subdir, linewidth=1)

    plt.title('VOUT vs. Time Plot')
    plt.xlabel('Time')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'Trans.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


root_dir = '/Users/hanwu/Downloads/Log_N65/Lab/b8cb5/Select_Points'
highlight_subdir = 'tmp_20240628192613852533942'
scan_and_process(root_dir, highlight_subdir)


def fourier_analysis(time_data, signal_data):
    sample_intervals = np.diff(time_data)
    sample_rate = 1 / np.mean(sample_intervals)

    fft_result = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(len(signal_data), 1 / sample_rate)

    positive_freqs = fft_freq > 0
    fft_magnitude = np.abs(fft_result[positive_freqs])
    fft_frequencies = fft_freq[positive_freqs]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_data, signal_data)
    plt.title('Time Series')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.stem(fft_frequencies, fft_magnitude, 'b', markerfmt=" ", basefmt="-b")
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sample_rate / 2)

    plt.tight_layout()
    plt.show()


# dict = extractTransTrace("/Users/hanwu/Downloads/Joblib/SSF_UM/tmp_20240411051043339987954/
# Trans.raw/tran.tran.tran.encode")
# print(dict)
# time_undershoot_index = [i for i, t in enumerate(dict['time']) if 5e-6 <= t <= 11e-6]
# time_undershoot = [dict['time'][i] for i in time_undershoot_index]
# vout_undershoot = [dict['VOUT'][i] for i in time_undershoot_index]
# fourier_analysis(time_undershoot, vout_undershoot)
