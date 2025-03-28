# from extract_trace import extractTransTrace_psf
from util.extract_trace import extractTransTrace_psf
import numpy as np
import bisect

def find_indices_in_range(nums, range_start, range_end):
    start_idx = bisect.bisect_left(nums, range_start)
    end_idx = bisect.bisect_right(nums, range_end)

    indices = list(range(start_idx, end_idx))

    return indices

def findTransLineReg_AXS(filename):
    try:
        trans_dict = extractTransTrace_psf(filename)
        time_series = trans_dict["time"]

        output_voltage_label = "net3"
        vout_trace = trans_dict[output_voltage_label]
        # print(f"vout_trace is {vout_trace}")

        time_ranges = (0.5e-3, 1e-3)
        time_indices = find_indices_in_range(time_series, time_ranges[0], time_ranges[1])
        clip_vout_trace = [vout_trace[i] for i in time_indices]

        vout_max = max(clip_vout_trace)
        vout_min = min(clip_vout_trace)
        vout_mean = np.mean(clip_vout_trace)
        # print(f"vout_max is {vout_max}, vout_min is {vout_min}, vout_mean is {vout_mean}")

        deltaVout = vout_max - vout_min

        if vout_mean <= 1.1 or vout_mean >= 1.3:
            deltaVout = 100.0
    except Exception as e:
        print(f"Error in findTransLineReg_AXS: {e}")
        deltaVout = 100.0

    return {"deltaVout": deltaVout}

# if __name__ == "__main__":
#     filename = "/Users/hanwu/Downloads/Netlist_AXS/Line_Reg_Trans.raw/tran.tran.tran"
#     print(findTransLineReg_AXS(filename))