from extract_trace import extractACTrace
import math
import os
import matplotlib.pyplot as plt


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'PSR.raw', 'ac.ac.encode')
        if os.path.isfile(target_file):
            trace_data = extractACTrace(target_file)

            trace_data['VOUT'] = [-20 * math.log10(vout) if vout != 0 else 0 for vout in trace_data['VOUT']]
            print(trace_data)
            results[subdir] = {
                'freq': trace_data['freq'],
                'VOUT': trace_data['VOUT']
            }
            print(results[subdir])

    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            plt.semilogx(data['freq'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['ILOAD'], data['VOUT'], label=subdir, color='blue', linewidth=1)
            plt.semilogx(data['freq'], data['VOUT'], label=subdir, linewidth=1)

    plt.title('PSR w/ 100mA Load')
    plt.xlabel('freq')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'PSR.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


root_dir = '/Users/hanwu/Downloads/Log_N65/Lab/b8cb5/Select_Points'
highlight_subdir = 'tmp_20240628192613852533942'
scan_and_process(root_dir, highlight_subdir)
