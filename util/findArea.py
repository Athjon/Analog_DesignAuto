import os
import re
from util.util_func import unit_conversion
# from util_func import unit_conversion


def findArea(filepath):
    """
    Calculate total transistor area from netlist file.

    Args:
    - filepath (str): Path to the simulation result file
    - tech_factor (float): Technology-dependent factor for area calculation, default is 17

    Returns:
    - dict: Dictionary with key 'area' and calculated area as value

    Example filepath: /home/wuhan/tb/train.tran
    Will look for first .scs file in /home/wuhan/tb/
    """
    try:
        # Get parent directory of input file
        parent_dir = os.path.dirname(os.path.dirname(filepath))

        # Find first .scs file in parent directory
        scs_file = None
        for file in os.listdir(parent_dir):
            if file.endswith('.scs'):
                scs_file = os.path.join(parent_dir, file)
                break

        if not scs_file:
            print(f"Warning: No .scs file found in {parent_dir}")
            return {"area": 0.0}

        # Read the scs file and find parameters line
        with open(scs_file, 'r') as f:
            for line in f:
                if line.startswith('parameters'):
                    params_line = line.strip()
                    break
            else:
                print("Warning: No parameters line found in scs file")
                return {"area": 0.0}

        # Extract required parameters using regex
        params = {}
        required_params = ['w_M0_per_finger', 'w_M1_per_finger', 'nf_M0', 'nf_M1']

        for param in required_params:
            pattern = rf'{param}=([0-9.]+[munp]?)'
            if param.startswith('nf'):
                pattern = rf'{param}=([0-9]+)'

            match = re.search(pattern, params_line)
            if not match:
                print(f"Warning: Parameter {param} not found")
                return {"area": 0.0}

            value = match.group(1)
            if param.startswith('nf'):
                params[param] = int(value)
            else:
                params[param] = unit_conversion(value)

        # Calculate total area
        area = params['w_M0_per_finger'] * params['nf_M0'] * 102 + params['w_M1_per_finger'] * params['nf_M1'] * 51

        return {"area": area}

    except Exception as e:
        print(f"Warning: Error calculating area: {str(e)}")
        return {"area": 0.0}


if __name__ == '__main__':
    # Test code
    filepath = "/Users/hanwu/Downloads/tb_Efficiency/tb_Efficiency.raw/tran.tran.tran"
    result = findArea(filepath)
    print(result)