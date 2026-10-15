A123 26650 battery parameter text files
========================================

All data files in this directory contain numeric values only: no header rows,
comments, labels, or missing-value placeholders. Values are written with 17
significant digits so IEEE-754 double precision values round-trip through
Python float parsing. Files are tab-delimited and can be read with
numpy.loadtxt(path, delimiter="\t").

Electrical lookup data
----------------------

ocv_c20_mean_v.txt
  Two columns: SOC [0..1], mean C/20 open-circuit voltage [V]. This is the
  V_avgC20 profile used by the bundled Simulink model.

ocv_c20_charge_v.txt, ocv_c20_discharge_v.txt
  Two columns: SOC [0..1], source charge/discharge C/20 voltage [V]. These are
  retained source profiles; the Simulink OCV lookup uses the mean profile.

entropic_coefficient_mV_per_K.txt
  Two columns: SOC, dU/dT [mV/K]. There are 11 source samples. The bundled
  Simulink model converts the coefficient to V/K with a 1/1000 gain. The source
  does not provide a temperature axis for this coefficient.

Each of the following ten files is a long-form lookup map with three columns:
SOC, temperature [deg C], parameter value. Rows are ordered by ascending SOC,
then ascending temperature, matching the source 9-by-8 matrices in row-major
order. Resistances are in ohm; capacitances are in farad. Charge and discharge
use separate SOC grids, but share the source temperature grid.

  rs_charge_ohm.txt       rs_discharge_ohm.txt
  r1_charge_ohm.txt        r1_discharge_ohm.txt
  r2_charge_ohm.txt        r2_discharge_ohm.txt
  c1_charge_f.txt          c1_discharge_f.txt
  c2_charge_f.txt          c2_discharge_f.txt

For each direction, the SOC and temperature coordinates are repeated on every
row so the file can be loaded directly as a numeric array. The source grids are:
  SOC charge:    0.1345 .. 0.93003 (9 points)
  SOC discharge: 0.105 .. 0.90053 (9 points)
  Temperature:  -20 .. 44.1169 deg C (8 points)
The original paper describes polynomial/Arrhenius fits, but the provided
package does not contain their fitted coefficient arrays. No new fit has been
made; these files contain the author-provided lookup values.

Thermal parameters
------------------

Each scalar file contains one value. Units are encoded in the filename:
  cc_j_per_k.txt                  core heat capacity [J/K], 62.7
  cs_j_per_k.txt                  surface heat capacity [J/K], 4.5
  rc_k_per_w.txt                  core-to-surface thermal resistance [K/W], 1.94
  ru_lin2014_k_per_w.txt          Lin et al. (2014) Table 1 Ru [K/W], 3.19
  ru_run_model_k_per_w.txt        bundled run_model.m Ru [K/W], 15
  *_95pct_ci_*.txt                lower and upper 95% confidence bounds

The paper's Ru value and the bundled run_model.m value differ; both are kept
under distinct filenames. No confidence interval for Cs is reported.

Cell specifications
-------------------

  nominal_capacity_ah.txt         bundled model capacity C_bat [Ah], 2.3
  measured_capacity_ah.txt        paper measured capacity [Ah], 2.1
  voltage_max_v.txt               maximum voltage [V], 3.6
  voltage_min_v.txt               minimum voltage [V], 2.0
  paper_max_tested_c_rate.txt     maximum paper validation test rate [C], 22

The 22C value is a tested operating condition, not a rated cell current.
The supplied files/paper do not specify nominal voltage or maximum rated charge
and discharge currents, so no files are created for those missing data.

Sources
-------

Author package: V_avgC20.mat, dUdT.mat, r_rc_rc_scale_cap_A123_26650.mat,
Battery_Electrothermal_Model.mdl, and run_model.m under
../Battery_Electrothermal_Model/. Cell/thermal values are cross-referenced to
Lin et al., “A Lumped-Parameter Electro-Thermal Model for Cylindrical
Batteries,” Journal of Power Sources 257 (2014), 1-11.
