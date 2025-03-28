from util.extract_trace import extractTrace
# from extract_trace import extractTrace

def findLoadReg_general(iload_name, vout_name, filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    # iload_name = '"ILOAD"'
    # vout_name = '"VOUT"'

    trace_dict = extractTrace(filename)

    iload_trace = trace_dict[iload_name]
    vout_trace = trace_dict[vout_name]

    iload_init = iload_trace[0]
    iload_end = iload_trace[-1]
    vout_init = vout_trace[0]
    vout_end = vout_trace[-1]

    try:
        load_reg = (vout_end - vout_init) / (iload_end - iload_init)
        load_reg = abs(load_reg)
    except ZeroDivisionError:
        load_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"loadReg": load_reg}

def findLoadReg(filename):
    iload_name = '"ILOAD"'
    vout_name = '"VOUT"'
    return findLoadReg_general(iload_name, vout_name, filename)

def findLoadReg_Haoqiang(filename):
    iload_name = '"iload"'
    vout_name = '"VOUT_OS"'
    return findLoadReg_general(iload_name, vout_name, filename)

def findLoadReg_AXS(filename):
    """
    Extract the trace data from a file and return the trace data.

    Args:
    - filename: Path to the file to be processed.
    - trace_name: Name of the trace to be extracted.
    """
    vout_name = '"net3"'

    trace_dict = extractTrace(filename)
    vout_trace = trace_dict[vout_name]

    iload_init = 0.001
    iload_end = 0.02
    vout_init = vout_trace[0]
    vout_end = vout_trace[-1]

    # print(f"vout_init is {vout_init}, vout_end is {vout_end}")

    try:
        load_reg = (vout_end - vout_init) / (iload_end - iload_init)
        load_reg = abs(load_reg)
    except ZeroDivisionError:
        load_reg = 100.0
        print("Warning: Division by zero. Setting load regulation to 100.0.")

    return {"loadReg": load_reg}

# if __name__ == "__main__":
#     filepath = "/Users/hanwu/Downloads/Netlist_AXS/Load_Reg.raw/dc.dc"
#     print(findLoadReg_AXS(filepath))