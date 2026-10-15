EPA Generic IPM 150 kW EDU motor parameter text files
=====================================================

Every parameter file contains numeric values only: no header, labels, or comments. Files are tab-delimited and written with 17 significant digits for IEEE-754 double round-trip. Read with numpy.loadtxt(path, delimiter="\t"). Each parameter/map has its own file. Scalars contain one value; vectors contain one value per line; lookup matrices retain the dimensions shown in the index.
NaN is used only to preserve a blank source cell or an invalid/undefined lookup value. Use the associated mask when querying a map.

SOURCE AND PHYSICAL SCOPE
-------------------------
Source package: EPA NCAT, Generic IPM 150kW EDU - ALPHA Map Package, version 2023-04. It is a generic EDU map derived from averaged confidential production-BEV IPMSRM data and scaled to 150 kW, not a map for a named production motor. EPA defines the map as combined electric motor + inverter + gearing (9.5:1); motor-only maps and separate motor/inverter/gearbox losses are unavailable.
The final package contains no independent regenerative test data. EPA mirrors drive-quadrant losses into regen with empirical factor 1.05. Therefore regenerative lookup values are modeled from motoring losses, not independent measurements.

ALPHA LOOKUP AXES AND MATRIX ORIENTATION
----------------------------------------
Full ALPHA matrices are 16 rows x 19 columns. Row i corresponds to motor_speed_axis_rpm.txt (or motor_speed_axis_radps.txt); column j corresponds to motor_torque_axis_nm.txt. EDU output-shaft axes are separately provided in edu_speed_axis_* and edu_torque_axis_* files, converted by speed_out = speed_motor / 9.5 and torque_out = torque_motor x 9.5.
Use operating_mask.txt for the torque/speed/power operating envelope. efficiency_defined_mask.txt excludes zero mechanical-power cells where efficiency has no definition. efficiency_lookup_pct.txt has NaN outside defined cells. total_power_loss_lookup_w_valid_region.txt has NaN outside the operating envelope. total_power_loss_lookup_w_source_rectangle.txt retains the calculated abs(P_elec-P_mech) over the whole table rectangle; mask it before simulation.
Efficiency convention: motoring eta = 100*P_mech/P_elec; regeneration eta = 100*P_elec/P_mech (both powers are negative). P_mech = omega_motor x torque_motor. Combined EDU loss = abs(P_elec-P_mech). Original workbook-6 efficiency values at zero mechanical power are retained as source values but flagged undefined by the corresponding mask.

WORKBOOK 3b MOTORING LOSS INPUT
-------------------------------
source_input_total_loss_kw.txt is the original EPA workbook-3b loss input: 18 torque rows x 16 speed columns. Rows align with source_input_torque_motor_nm.txt; columns align with source_input_speed_motor_rpm.txt. Blank source cells are NaN and are marked in source_input_total_loss_valid_mask.txt. source_input_total_loss_w.txt is the same table converted from kW to W (x1000).

WORKBOOK 6 SAMPLE CHANNELS
--------------------------
Every source_sample_* file has 204 rows and aligned row indices. The original channels are motor speed [rpm], motor torque [N·m], mechanical power [kW], electrical power [kW], and efficiency [%]. EDU-shaft speed/torque and total loss are derived channels. The 204 point list includes motoring, regenerative, and zero-mechanical-power samples.

PARAMETER INTERPRETATION AND ABSENT DATA
----------------------------------------
max_motor_power_w.txt is the maximum power capacity specified by ALPHA (150 kW); the source does not state rated vs peak vs continuous duration. The ALPHA scalar maximum torque is 312 Nm; use positive_torque_limit_* and negative_torque_limit_* for the speed-dependent envelope. nominal_voltage_for_torque_limit_v.txt is the voltage used for the torque-limit lookup, not a declared rated terminal voltage. gear_efficiency_norm.txt = 1 is a model normalization field, not measured gearbox efficiency.
peak_*_efficiency files are map/sample peaks and are not rated efficiency. EPA does not provide rated power, continuous power, rated torque, rated speed, base speed, rated efficiency, or a rated speed/torque operating point; no files are made for those missing quantities.
motor_lifetime_h_user_assumption.txt is the user-provided Peng et al. (2024) assumption (30,000 h), not an EPA parameter. The map supports total-EDU energy-loss accumulation, but cannot isolate motor-only loss or supply a rated motor-loss reference for CLR/SOH normalization.

EXPORTED FILE INDEX
-------------------
File	Shape	Unit	Meaning
edu_speed_axis_radps.txt	16 x 1	rad/s	EDU output-shaft speed lookup axis; motor speed divided by gear ratio.
edu_speed_axis_rpm.txt	16 x 1	rpm	EDU output-shaft speed lookup axis; motor speed divided by gear ratio.
edu_torque_axis_nm.txt	19 x 1	N·m	EDU output-shaft torque lookup axis; motor torque multiplied by gear ratio.
efficiency_defined_map_node_count.txt	1 x 1	count	Count of full-map cells where efficiency is defined.
efficiency_defined_mask.txt	16 x 19	0/1	1 where efficiency is defined; zero mechanical power and invalid cells are excluded.
efficiency_lookup_pct.txt	16 x 19	%	Derived complete EDU efficiency lookup. Rows align with motor speed axis; columns align with motor torque axis. NaN marks undefined/out-of-domain cells.
electric_power_lookup_w.txt	16 x 19	W	EPA ALPHA electric-power table. Rows align with motor speed axis; columns with motor torque axis. Negative values indicate electrical power returned in regeneration.
gear_efficiency_norm.txt	1 x 1	dimensionless	EPA ALPHA gear-efficiency normalization parameter (1); not an independently measured gearbox efficiency.
gear_ratio.txt	1 x 1	dimensionless	Motor-to-EDU-output gear ratio.
inertia_kgm2.txt	1 x 1	kg·m²	Rotational inertia parameter specified in the EPA ALPHA model.
loss_component_breakdown_available.txt	1 x 1	0/1	Flag for availability of motor/inverter/gearbox loss breakdown (0 = unavailable).
max_generator_power_w.txt	1 x 1	W	Maximum generator/regenerative power magnitude in ALPHA.
max_motor_power_w.txt	1 x 1	W	Maximum motor power capacity in ALPHA; rated/continuous duration is unspecified.
max_motoring_torque_edu_nm.txt	1 x 1	N·m	Maximum positive torque envelope magnitude converted to EDU output shaft.
max_motoring_torque_motor_nm.txt	1 x 1	N·m	Maximum positive torque envelope magnitude on motor shaft.
max_regenerative_power_w.txt	1 x 1	W	Maximum regenerative power magnitude, same limit as maximum generator power.
max_regenerative_torque_edu_nm.txt	1 x 1	N·m	Maximum negative torque envelope magnitude converted to EDU output shaft.
max_regenerative_torque_motor_nm.txt	1 x 1	N·m	Maximum negative torque envelope magnitude on motor shaft.
max_speed_edu_radps.txt	1 x 1	rad/s	Maximum EDU output-shaft speed converted from motor speed.
max_speed_edu_rpm.txt	1 x 1	rpm	Maximum EDU output-shaft speed converted from motor speed.
max_speed_motor_radps.txt	1 x 1	rad/s	Maximum motor-shaft speed specified by ALPHA.
max_speed_motor_rpm.txt	1 x 1	rpm	Maximum motor-shaft speed converted from rad/s.
max_torque_edu_nm.txt	1 x 1	N·m	ALPHA scalar maximum motor torque converted to EDU output shaft.
max_torque_motor_nm.txt	1 x 1	N·m	ALPHA scalar maximum motor torque (312 Nm); the detailed speed envelope is in torque-limit files.
mechanical_power_lookup_w.txt	16 x 19	W	Derived motor mechanical power = angular speed x torque. Rows are motor speed; columns are motor torque.
motor_lifetime_h_user_assumption.txt	1 x 1	h	User-provided Peng et al. (2024) lifetime assumption; not an EPA value.
motor_only_map_available.txt	1 x 1	0/1	Flag for a motor-only map (0 = unavailable; this map is combined EDU).
motor_speed_axis_radps.txt	16 x 1	rad/s	Motor-shaft speed breakpoints for ALPHA maps.
motor_speed_axis_rpm.txt	16 x 1	rpm	Motor-shaft speed breakpoints for ALPHA maps.
motor_torque_axis_nm.txt	19 x 1	N·m	Motor-shaft torque breakpoints for ALPHA maps.
motoring_map_node_count.txt	1 x 1	count	Count of valid motoring-quadrant cells with defined efficiency.
negative_torque_limit_edu_nm.txt	16 x 1	N·m	Negative/regenerative torque limit at each speed, converted to EDU output shaft; row i aligns with speed-axis row i.
negative_torque_limit_motor_nm.txt	16 x 1	N·m	Negative/regenerative motor torque limit at each speed; row i aligns with speed-axis row i.
nominal_voltage_for_torque_limit_v.txt	1 x 1	V	Voltage input used for the torque-limit lookup; not stated as rated motor terminal voltage.
operating_map_node_count.txt	1 x 1	count	Count of grid cells inside modeled speed, torque, and power limits.
operating_mask.txt	16 x 19	0/1	1 inside modeled speed/torque/power envelope, else 0.
peak_model_grid_efficiency_pct.txt	1 x 1	%	Maximum derived efficiency on the valid lookup grid; not rated efficiency.
peak_sample_efficiency_pct.txt	1 x 1	%	Maximum efficiency among EPA sample points with nonzero mechanical power; not rated efficiency.
positive_torque_limit_edu_nm.txt	16 x 1	N·m	Positive/motoring torque limit at each speed, converted to EDU output shaft; row i aligns with speed-axis row i.
positive_torque_limit_motor_nm.txt	16 x 1	N·m	Positive/motoring motor torque limit at each speed; row i aligns with speed-axis row i.
regenerative_loss_mirror_factor.txt	1 x 1	dimensionless	EPA empirical factor used to mirror motoring losses into regeneration; no independent regen data were available.
regenerative_map_node_count.txt	1 x 1	count	Count of valid regenerative-quadrant cells with defined efficiency.
source_input_speed_motor_rpm.txt	16 x 1	rpm	Motor-speed axis from EPA workbook 3b derived-loss input.
source_input_torque_motor_nm.txt	18 x 1	N·m	Motor-torque axis from EPA workbook 3b derived-loss input.
source_input_total_loss_kw.txt	18 x 16	kW	Workbook-3b derived motoring loss table in original kW; rows torque, columns speed. NaN preserves blank cells.
source_input_total_loss_valid_mask.txt	18 x 16	0/1	1 for a populated source loss cell in workbook 3b; 0 for a blank cell.
source_input_total_loss_w.txt	18 x 16	W	Workbook-3b motoring loss table converted from kW to W (x1000); rows torque, columns speed.
source_point_count.txt	1 x 1	count	Numeric field sourcePointCount from motor_parameters.mat; see group notes and unit suffix.
source_sample_edu_speed_rpm.txt	204 x 1	rpm	Derived output speed = motor speed / gear ratio.
source_sample_edu_torque_nm.txt	204 x 1	N·m	Derived output torque = motor torque x gear ratio.
source_sample_efficiency_defined_mask.txt	204 x 1	0/1	1 where source efficiency is defined (nonzero mechanical power); else 0.
source_sample_efficiency_pct.txt	204 x 1	%	Original EPA workbook-6 efficiency. Source stores 0 at zero mechanical power; use the defined mask.
source_sample_electrical_power_kw.txt	204 x 1	kW	Original EPA workbook-6 electrical power.
source_sample_mechanical_power_kw.txt	204 x 1	kW	Original EPA workbook-6 mechanical power.
source_sample_motor_speed_rpm.txt	204 x 1	rpm	Original EPA workbook-6 motor speed; row-aligned with all source_sample_* channels.
source_sample_motor_torque_nm.txt	204 x 1	N·m	Original EPA workbook-6 motor torque; row-aligned with all source_sample_* channels.
source_sample_motoring_mask.txt	204 x 1	0/1	1 for positive mechanical-power motoring points; else 0.
source_sample_regenerative_mask.txt	204 x 1	0/1	1 for negative mechanical-power regenerative points; else 0.
source_sample_total_loss_kw.txt	204 x 1	kW	Derived total loss = abs(mechanical power - electrical power), using workbook-6 channels.
source_sample_zero_mechanical_power_mask.txt	204 x 1	0/1	1 where mechanical power is zero; efficiency is undefined there.
total_power_loss_lookup_w_source_rectangle.txt	16 x 19	W	Derived total EDU loss abs(P_elec-P_mech) on the full ALPHA rectangle, including cells outside limits; mask before simulation.
total_power_loss_lookup_w_valid_region.txt	16 x 19	W	Derived total EDU loss abs(P_elec-P_mech) inside modeled operating envelope; NaN outside.

SOURCE FILES
------------
3a- Generic IPM 150kW EDU - ALPHA Generation Process.html; 3b- Generic IPM 150kW EDU - Derived Input Data File.xlsx; 6- Generic IPM 150kW EDU - Electrical Power Consumption Data.xlsx; 7- emachine_generic_IPM_150kW_EDU.m under ../150kw_EDU/. Source MAT: ../150kw_EDU/motor_parameters.mat. Source files were read only.
Conversions: rpm = rad/s x 60/(2*pi); EDU speed = motor speed / 9.5; EDU torque = motor torque x 9.5; W = kW x 1000.
