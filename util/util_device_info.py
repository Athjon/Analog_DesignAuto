import re
import csv


def parse_device_info(filepath):
    def extract_device_properties(device_section):
        # Extract properties for each device with a flexible approach
        property_patterns = [
            (r'\"(.+?)\" FLOAT DOUBLE PROP\(\n(\"units\" \"(.+?)\"\n)?\"description\" \"(.+?)\"\n\)', 'FLOAT DOUBLE'),
            (r'\"(.+?)\" INT BYTE PROP\(\n\"description\" \"(.+?)\"\n\)', 'INT BYTE')
        ]

        properties = []
        for pattern, dtype in property_patterns:
            property_matches = re.findall(pattern, device_section)
            for match in property_matches:
                if dtype == 'FLOAT DOUBLE':
                    prop_name, _, prop_unit, prop_description = match
                elif dtype == 'INT BYTE':
                    prop_name, prop_description = match
                    prop_unit = None
                property_info = {
                    'property_name': prop_name.strip(),
                    'property_type': dtype,
                    'property_unit': prop_unit.strip() if prop_unit else None,
                    'property_description': prop_description.strip()
                }
                properties.append(property_info)

        return properties

    # Read the file content
    with open(filepath, "r") as file:
        file_content = file.read()

    # Check if the required section exists
    match = re.search(r'TYPE(.+?)VALUE', file_content, re.DOTALL)
    if not match:
        raise ValueError("The content between 'TYPE' and 'VALUE' not found in the file.")

    type_value_section = match.group(1)

    # Split the section into individual device sections based on the provided delimiter
    device_sections = type_value_section.split('PROP(\n"key" "inst"\n)\n')

    # Filter out empty or invalid sections and parse each device section
    parsed_devices = []
    for section in device_sections:
        if section.strip() and '" STRUCT(' in section:
            device_name_match = re.search(r'\"(\w+)\" STRUCT', section)
            device_name = device_name_match.group(1) if device_name_match else None
            properties = extract_device_properties(section)
            device_info = {
                'device_name': device_name,
                'device_type': 'STRUCT',
                'properties': properties
            }
            parsed_devices.append(device_info)
    return parsed_devices


def device_info_export(device_info, output_filepath):
    # CSV headers
    headers = ["DeviceName", "DeviceType", "ParameterName", "ParameterDateType", "ParameterUnit",
               "ParameterDescription"]

    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for device in device_info:
            device_name = device['device_name']
            device_type = device['device_type']
            for prop in device['properties']:
                row = [
                    device_name,
                    device_type,
                    prop['property_name'],
                    prop['property_type'],
                    prop['property_unit'],
                    prop['property_description']
                ]
                writer.writerow(row)


# Testing the function with the provided content
# test_input_filepath = ("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/"
#                        "netlist_assign_test/DC.raw/dcOpInfo.info.encode")
# test_output_filepath = ("/Users/hanwu/ML/AnalogDesignAuto_MultiAgent/custom_env/"
#                         "netlist_assign_test/DC.raw/dcOpInfo.info.encode.csv")
#
# test_parsed_devices_info = parse_device_info(test_input_filepath)
# print(test_parsed_devices_info)
# device_info_export(test_parsed_devices_info, test_output_filepath)

# Output
# [{'device_name': 'bsim4', 'device_type': 'STRUCT', 'properties': [{'property_name': 'ids', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Resistive drain-to-source current'},
# {'property_name': 'vgs', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description':
# 'Gate-source voltage'}, {'property_name': 'vds', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V',
# 'property_description': 'Drain-source voltage'}, {'property_name': 'vbs', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'V', 'property_description': 'Bulk-source voltage'}, {'property_name': 'vgd', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description': 'Gate-drain voltage'}, {'property_name': 'vdb',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description': 'Drain-bulk voltage'},
# {'property_name': 'vgb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description': 'Gate-bulk
# voltage'}, {'property_name': 'vth', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description':
# 'Threshold voltage (alias=lv9)'}, {'property_name': 'vdsat', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V',
# 'property_description': 'Drain-source saturation voltage (alias=lv10)'}, {'property_name': 'gm', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'S', 'property_description': 'Common-source transconductance (alias=lx7)'},
# {'property_name': 'gds', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'S', 'property_description':
# 'Common-source output conductance (alias=lx8)'}, {'property_name': 'gmbs', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'S', 'property_description': 'Body-transconductance (alias=lx9)'}, {'property_name': 'betaeff',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A/V^2', 'property_description': 'Effective beta (alias LV21)'},
# {'property_name': 'cjd', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Drain-bulk
# junction capacitance (alias=lx29)'}, {'property_name': 'cjs', 'property_type': 'FLOAT DOUBLE', 'property_unit':
# 'F', 'property_description': 'Source-bulk junction capacitance (alias=lx28)'}, {'property_name': 'cgg',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Total gate capacitance,
# including intrinsic, overlap and fringing components (alias=lx82)'}, {'property_name': 'cgd', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Total gate-to-drain capacitance, including
# intrinsic, overlap, and fringing components (alias=lx83)'}, {'property_name': 'cgs', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'F', 'property_description': 'Total gate-to-source capacitance, including intrinsic,
# overlap, and fringing components (alias=lx84)'}, {'property_name': 'cgb', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'Total gate-to-bulk capacitance, including intrinsic and overlap
# components'}, {'property_name': 'cdg', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'Total drain-to-gate capacitance, including intrinsic, overlap, and fringing components (
# alias=lx87)'}, {'property_name': 'cdd', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'Drain capacitance, including intrinsic, overlap, and fringing components'},
# {'property_name': 'cds', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Total
# drain-to-source capacitance (alias=lx86)'}, {'property_name': 'cdb', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'Intrinsic drain-to-bulk capacitance'}, {'property_name': 'csg',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Total source-to-gate capacitance,
# including intrinsic, overlap, and fringing components'}, {'property_name': 'csd', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'Total source-to-drain capacitance'}, {'property_name': 'css',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Source capacitance,
# including intrinsic, overlap, and fringing components'}, {'property_name': 'csb', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'Intrinsic source-to-bulk capacitance'}, {'property_name': 'cbg',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Total bulk-to-gate capacitance,
# including intrinsic and overlap components (alias=lx88)'}, {'property_name': 'cbd', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'F', 'property_description': 'Intrinsic bulk-to-drain capacitance'}, {'property_name':
# 'cbs', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'Intrinsic bulk-to-source
# capacitance'}, {'property_name': 'cbb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'Bulk capacitance, including intrinsic and overlap components'}, {'property_name':
# 'covlgs', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F/m', 'property_description': 'Gate-source overlap and
# fringing capacitances (alias=lv36)'}, {'property_name': 'covlgd', 'property_type': 'FLOAT DOUBLE', 'property_unit':
# 'F/m', 'property_description': 'Gate-drain overlap and fringing capacitances (alias=lv37)'}, {'property_name':
# 'covlgb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F/m', 'property_description': 'Gate-bulk overlap
# capacitances (alias=lv38)'}, {'property_name': 'cggbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'CGGBO = dQg/dVg intrinsic gate capacitance (alias=lx18)'}, {'property_name': 'cgdbo',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'CGDBO = -dQg/dVd intrinsic
# gate-to-drain capacitance (alias=lx19)'}, {'property_name': 'cgsbo', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'CGSBO = -dQg/dVs intrinsic gate-to-source capacitance (
# alias=lx20)'}, {'property_name': 'cbgbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'CBGBO = -dQb/dVg intrinsic bulk-to-gate capacitance (alias=lx21)'}, {'property_name':
# 'cbdbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'CBDBO = -dQb/dVd intrinsic
# bulk-to-drain capacitance (alias=lx22)'}, {'property_name': 'cbsbo', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'F', 'property_description': 'CBSBO = -dQb/dVs intrinsic bulk-to-source capacitance (
# alias=lx23)'}, {'property_name': 'cdgbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'CDGBO = -dQd/dVg intrinsic drain-to-gate capacitance (alias=lx32)'}, {'property_name':
# 'cddbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F', 'property_description': 'CDDBO = dQd/dVd intrinsic
# drain capacitance (alias=lx33)'}, {'property_name': 'cdsbo', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'F',
# 'property_description': 'CDSBO = -dQd/dVs intrinsic drain-to-source capacitance (alias=lx34)'}, {'property_name':
# 'ron', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Ohm', 'property_description': 'On-resistance'},
# {'property_name': 'id', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Resistive
# drain current'}, {'property_name': 'ig', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A',
# 'property_description': 'Total DC gate current'}, {'property_name': 'is', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'A', 'property_description': 'Total DC source current'}, {'property_name': 'ib', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Total DC bulk current'}, {'property_name': 'ibulk',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Resistive bulk current'},
# {'property_name': 'pwr', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'W', 'property_description': 'Power at
# op point'}, {'property_name': 'gmoverid', 'property_type': 'FLOAT DOUBLE', 'property_unit': '1/V',
# 'property_description': 'Gm/Ids'}, {'property_name': 'ueff', 'property_type': 'FLOAT DOUBLE', 'property_unit':
# None, 'property_description': 'ueff'}, {'property_name': 'rdeff', 'property_type': 'FLOAT DOUBLE', 'property_unit':
# 'Ohm', 'property_description': 'Effective drain resistance'}, {'property_name': 'rseff', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'Ohm', 'property_description': 'Effective source resistance'}, {'property_name': 'rgbd',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Ohm', 'property_description': 'Gate bias-dependent resistance'},
# {'property_name': 'igidl', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description':
# 'Gate-induced drain leakage current (alias=lx70)'}, {'property_name': 'igisl', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'A', 'property_description': 'Gate-induced source leakage current'}, {'property_name': 'igdt',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Gate Dielectric tunneling current (
# alias=lx71)'}, {'property_name': 'igd', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A',
# 'property_description': 'Gate-to-drain tunneling current (alias=lx39)'}, {'property_name': 'igs', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Gate-to-source tunneling current (alias=lx38)'},
# {'property_name': 'igb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description':
# 'Gate-to-bulk tunneling current (alias=lx66)'}, {'property_name': 'igbacc', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'A', 'property_description': 'Gate-to-bulk tunneling current determined by ECB (alias=lx73)'},
# {'property_name': 'igbinv', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description':
# 'Gate-to-bulk tunneling current determined by EVB (alias=lx74)'}, {'property_name': 'igcs', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'A', 'property_description': 'Gate-to-channel (source side) tunneling current (
# alias=lx67)'}, {'property_name': 'igcd', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A',
# 'property_description': 'Gate-to-channel (drain side) tunneling current (alias=lx68)'}, {'property_name': 'isub',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Substrate current ( alias to LX69
# )'}, {'property_name': 'gbs', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'S', 'property_description':
# 'Bulk-source diode conductance (alias=lx11)'}, {'property_name': 'gbd', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'S', 'property_description': 'Bulk-drain diode conductance (alias=lx10)'}, {'property_name': 'qg',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description': 'Total gate charge,
# including intrinsic, overlap and fringing components'}, {'property_name': 'qd', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'Coul', 'property_description': 'Total drain charge, including intrinsic, overlap and fringing
# components'}, {'property_name': 'qs', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul',
# 'property_description': 'Total source charge, including intrinsic, overlap and fringing components'},
# {'property_name': 'qb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description': 'Total
# bulk charge, including intrinsic and overlap components'}, {'property_name': 'qjd', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'Coul', 'property_description': 'Drain-bulk junction charge'}, {'property_name': 'qjs',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description': 'Source-bulk junction charge'},
# {'property_name': 'qgdovl', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description':
# 'Gate-drain overlap and fringing charge'}, {'property_name': 'qgsovl', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'Coul', 'property_description': 'Gate-source overlap and fringing charge'}, {'property_name':
# 'qgi', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description': 'Intrinsic gate charge'},
# {'property_name': 'qdi', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul', 'property_description':
# 'Intrinsic drain charge'}, {'property_name': 'qsi', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Coul',
# 'property_description': 'Intrinsic source charge'}, {'property_name': 'qbi', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'Coul', 'property_description': 'Intrinsic bulk charge'}, {'property_name': 'ide',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Total DC drain current'},
# {'property_name': 'ige', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Total DC
# gate current'}, {'property_name': 'ise', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A',
# 'property_description': 'Total DC source current'}, {'property_name': 'ibe', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'A', 'property_description': 'Total DC bulk current'}, {'property_name': 'idb', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'DC drain-bulk current'}, {'property_name': 'isb',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'DC source-bulk current'},
# {'property_name': 'vsb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description':
# 'Source-bulk DC voltage'}, {'property_name': 'gmb', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'S',
# 'property_description': 'DC bulk transconductance'}, {'property_name': 'vgt', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'V', 'property_description': 'effective gate drive voltage including back bias and drain bias
# effects'}, {'property_name': 'vth_drive', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V',
# 'property_description': 'effective gate drive voltage including back bias and drain bias effects. Alias for Vgt'},
# {'property_name': 'vdss', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description': 'Drain
# saturation voltage at actual bias'}, {'property_name': 'vsat_marg', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'V', 'property_description': 'Vds margin'}, {'property_name': 'vdsat_marg', 'property_type':
# 'FLOAT DOUBLE', 'property_unit': 'V', 'property_description': 'Vds margin'}, {'property_name': 'self_gain',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': None, 'property_description': 'Transistor self gain'},
# {'property_name': 'rout', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Ohm', 'property_description': 'AC
# output resistor'}, {'property_name': 'beff', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A/V^2',
# 'property_description': 'Gain factor in saturation'}, {'property_name': 'fug', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'Hz', 'property_description': 'Unity current gain frequency at actual bias'}, {'property_name':
# 'ft', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Hz', 'property_description': 'Unity current gain frequency
# at actual bias'}, {'property_name': 'rgate', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'Ohm',
# 'property_description': 'MOS gate resistance'}, {'property_name': 'vearly', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'V', 'property_description': 'Equivalent early voltage'}, {'property_name': 'region',
# 'property_type': 'INT BYTE', 'property_unit': None, 'property_description': 'Estimated operating region. %Z outputs
# the number (0-4) in a rawfile'}, {'property_name': 'reversed', 'property_type': 'INT BYTE', 'property_unit': None,
# 'property_description': 'Reverse mode indicator'}]}, {'device_name': 'isource', 'device_type': 'STRUCT',
# 'properties': [{'property_name': 'i', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A',
# 'property_description': 'Current through the source'}, {'property_name': 'v', 'property_type': 'FLOAT DOUBLE',
# 'property_unit': 'V', 'property_description': 'Voltage across the source'}, {'property_name': 'pwr',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'W', 'property_description': 'Power dissipation'}]},
# {'device_name': 'vsource', 'device_type': 'STRUCT', 'properties': [{'property_name': 'v', 'property_type': 'FLOAT
# DOUBLE', 'property_unit': 'V', 'property_description': 'Voltage across the source'}, {'property_name': 'i',
# 'property_type': 'FLOAT DOUBLE', 'property_unit': 'A', 'property_description': 'Current through the source'},
# {'property_name': 'pwr', 'property_type': 'FLOAT DOUBLE', 'property_unit': 'W', 'property_description': 'Power
# dissipation'}]}]

