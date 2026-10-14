# EPA Generic IPM 150 kW EDU map

This directory contains the EPA ALPHA map package released in April 2023 and a MATLAB reconstruction of its published efficiency-map sample points.

## Source package

- Official EPA package page: https://www.epa.gov/vehicle-and-fuel-emissions-testing/combining-data-complete-emotor-alpha-maps
- Original package ZIP: `../EPA_Generic_IPM_150kW_EDU_ALPHA_Map_Package_2023-04.zip`
- Suggested citation from the package: *Generic IPM 150kW EDU – ALPHA Map Package*. Version 2023-04. Ann Arbor, MI: US EPA, National Vehicle and Fuel Emissions Laboratory, National Center for Advanced Technology, 2023.
- The numeric source used for the plot is `6- Generic IPM 150kW EDU - Electrical Power Consumption Data.xlsx`. Its columns contain motor-shaft speed (rpm), motor-shaft torque (N·m), mechanical power (kW), electrical power (kW), and efficiency (%).

## Processed files

- `EPA_Generic_IPM_150kW_EDU_efficiency_map.mat` contains the original sample points, converted EDU output-shaft coordinates, and the interpolated grid used for plotting.
- `EPA_Generic_IPM_150kW_EDU_efficiency_map_points.txt` contains the 204 numeric sample rows, including zero-power-axis points, as a tab-delimited table.
- Figures are saved in the repository `results/` directory. The editable MATLAB `.fig` files and plotting script are in `matlab_simulink/`.

The package specifies a 9.5:1 gear ratio. For the plotted EDU output-shaft axes, speed is converted as motor speed / 9.5 and torque as motor torque × 9.5. The efficiency values come from the EPA sample data and are not rescaled. The 0 rpm and 0 N·m axes are masked in the figures because the source assigns zero efficiency where mechanical output power is zero; the source rows remain in the `.mat` and `.txt` files.

The surface is a linear scattered-data interpolation of 204 points extracted from the EPA efficiency map, constrained by the sample-derived torque envelope and 150 kW power limit. It is a reconstruction of the published sample data, not EPA's full-resolution internal map or a new Simulink simulation. EPA states that the generic map was derived from averaged confidential benchmarking data and represents the motor, inverter, and gearing together.
