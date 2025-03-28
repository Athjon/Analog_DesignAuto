from util.extract_trace import extractTrace
# from extract_trace import extractTrace


def findLineReg(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    vdd_name = '"VIN"'
    vout_name = '"VOUT"'

    trace_dict = extractTrace(filename)

    vdd_trace = trace_dict[vdd_name]
    vout_trace = trace_dict[vout_name]

    vdd_init = vdd_trace[0]
    vdd_end = vdd_trace[-1]
    vout_init = vout_trace[0]
    vout_end = vout_trace[-1]

    try:
        line_reg = (vout_end - vout_init) / (vdd_end - vdd_init)
        line_reg = abs(line_reg)
    except ZeroDivisionError:
        line_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"lineReg": line_reg}


def findLineReg_AXS(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    vdd_name = '"VDDI"'
    vout_name = '"net3"'

    trace_dict = extractTrace(filename)

    vdd_trace = trace_dict[vdd_name]
    vout_trace = trace_dict[vout_name]

    vdd_init = vdd_trace[0]
    vdd_end = vdd_trace[-1]
    # vout_init is the min value of vout_trace, vout_end is the max value of vout_trace
    vout_init = min(vout_trace)
    vout_end = max(vout_trace)
    # print(f"vdd_init is {vdd_init}, vdd_end is {vdd_end}, vout_init is {vout_init}, vout_end is {vout_end}")

    try:
        line_reg = (vout_end - vout_init) / (vdd_end - vdd_init)
        line_reg = abs(line_reg)
    except ZeroDivisionError:
        line_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"lineReg": line_reg}

# if __name__ == "__main__":
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/Line_Reg.raw/dc.dc"
#     print(findLineReg_AXS(file_path))
