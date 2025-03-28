import logging

def handle_range_value(value):
    """
    Handle potential range values (lists) by converting them to single values

    Args:
        value: Either a single numeric value or a list of [min, max]
    Returns:
        float: Single numeric value. For lists, returns average of min and max
    """
    if isinstance(value, list):
        # For range type, use the average of min and max
        return sum(value) / len(value)
    return float(value)  # Convert to float for consistency


def norm_ideal_spec(input_specs, norm_specs):
    """
    Normalize the current specs by the ideal specs.
    Handle three types of values:
    - Single value for max/min objectives
    - List [min, max] for range objectives
    - Regular numeric values

    Args:
        input_specs: dict, ideal specs for the circuit
        norm_specs: dict, current specs for the circuit
    Returns:
        norm_cur_specs: dict, normalized specs for the circuit
    """
    norm_cur_specs = {}
    logging.debug(f'input_specs: {input_specs}')
    logging.debug(f'norm_specs: {norm_specs}')

    for key, sub_dict in norm_specs.items():
        norm_cur_specs_sub = {}
        for sub_key, value in sub_dict.items():
            # Get and process the values
            ideal_value = handle_range_value(value)
            input_value = handle_range_value(input_specs[sub_key]['value'])

            logging.debug(f'Processing {sub_key}:')
            logging.debug(f'  Original ideal_value: {value}')
            logging.debug(f'  Processed ideal_value: {ideal_value}')
            logging.debug(f'  Original input_value: {input_specs[sub_key]["value"]}')
            logging.debug(f'  Processed input_value: {input_value}')

            # Calculate normalized value
            norm_value = (input_value - ideal_value) / (input_value + ideal_value)
            logging.debug(f'  Normalized value: {norm_value}')

            norm_cur_specs_sub[sub_key] = norm_value
        norm_cur_specs[key] = norm_cur_specs_sub

    return norm_cur_specs


def norm_sim_spec(sim_specs, norm_specs):
    """
    Normalize simulation specs using the same approach as norm_ideal_spec

    Args:
        sim_specs: dict, simulation results
        norm_specs: dict, normalization specifications
    Returns:
        dict: Normalized simulation specs
    """
    norm_cur_specs = {}
    for outer_k in sim_specs:
        norm_cur_specs[outer_k] = {}
        for inner_k in sim_specs[outer_k]:
            sim_value = sim_specs[outer_k][inner_k]
            norm_value = handle_range_value(norm_specs[outer_k][inner_k])
            norm_cur_specs[outer_k][inner_k] = (sim_value - norm_value) / (sim_value + norm_value)

    return norm_cur_specs