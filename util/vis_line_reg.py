from extract_trace import extractTrace
import os
import matplotlib.pyplot as plt


def scan_and_process(root_dir, highlight_subdir):
    results = {}

    for subdir in next(os.walk(root_dir))[1]:
        target_file = os.path.join(root_dir, subdir, 'Line_Reg_1m.raw', 'dc.dc.encode')
        if os.path.isfile(target_file):
            trace_data = extractTrace(target_file)
            print(trace_data)
            results[subdir] = {
                'VIN': trace_data['"VIN"'],
                'VOUT': trace_data['"VOUT"']
            }

    plot_data(results, highlight_subdir, root_dir)


def plot_data(results, highlight_subdir, root_dir):
    plt.figure(figsize=(10, 6))

    for subdir, data in results.items():
        if subdir == highlight_subdir:
            try:
                plt.plot(data['VIN'][0::2], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
            except:
                plt.plot(data['VIN'], data['VOUT'], label=f'{subdir} (highlight)', color='red', linewidth=2)
        else:
            # plt.plot(data['VDD'][0::2], data['VOUT'], label=subdir, color='blue', linewidth=1)
            try:
                plt.plot(data['VIN'][0::2], data['VOUT'], label=subdir, linewidth=1)
            except:
                plt.plot(data['VIN'], data['VOUT'], label=subdir, linewidth=1)

    plt.title('VOUT vs. VIN Plot, Iload = 1mA')
    plt.xlabel('VIN')
    plt.ylabel('VOUT')
    plt.legend()
    plt.grid(True)

    plt_path = os.path.join(root_dir, 'Line_Reg_1m.png')
    plt.savefig(plt_path)
    plt.show()
    print(f"Image save to：{plt_path}")


root_dir = '/Users/hanwu/Downloads/Log_N65/Lab/b8cb5/Select_Points'
highlight_subdir = 'tmp_20240628192613852533942'
scan_and_process(root_dir, highlight_subdir)
