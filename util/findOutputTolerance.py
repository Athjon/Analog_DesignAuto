def extract_dcOP_data(file_path, search_keyword):
    """
    Search the first line in a file containing a specific keyword and extract float data.

    Args:
    - file_path: Path to the file to be processed.
    - search_keyword: Keyword to search for in the file.

    Returns:
    - A float extracted from the line containing the keyword, or None if not found.
    """
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if search_keyword in line:
                    # Split the line by space and extract the third element as data
                    parts = line.split()
                    if len(parts) >= 3:  # Ensure there are at least 3 parts
                        extracted_data = float(parts[2])  # Convert to float
                    break  # Stop searching after finding the first match
        if extracted_data is None:
            raise ValueError("Keyword not found or file does not contain valid data format.")
    except FileNotFoundError:
        raise FileNotFoundError("File does not exist.")
    except ValueError as e:
        raise ValueError(f"Error while processing the file: {e}")

    return abs(extracted_data)

def findOutputTolerance_AXS(filepath):
    search_keyword = "net6"
    try:
        value = extract_dcOP_data(filepath, search_keyword)
        tolerance = (value - 1.2) / 1.2
        tolerance = abs(tolerance)
    except Exception as e:
        print(f"Warning: {e}. Setting value to default (0.0).")
        tolerance = 100.0

    return {"outputTolerance": tolerance}

# if __name__ == '__main__':
#     file_path = "/Users/hanwu/Downloads/Netlist_AXS/Output_Tolerance_20m.raw/dcOp.dc"
#     print(findOutputTolerance_AXS(file_path))