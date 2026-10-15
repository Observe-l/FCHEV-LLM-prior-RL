# Lin and Zheng (2011) 50 kW FCS data

## Source and scope

W.-S. Lin and C.-H. Zheng, “Energy management of a fuel cell/ultracapacitor hybrid power system using an adaptive optimal-control method,” *Journal of Power Sources*, 196(6), 3280–3289 (2011), DOI: [10.1016/j.jpowsour.2010.11.127](https://doi.org/10.1016/j.jpowsour.2010.11.127). The local source copy is `../Battery_Electrothermal_Model/papers/Energy management of a fuel cell_ultracapacitor hybrid power system using an adaptive optimal-control method.pdf`.

The calculations below use the paper's original 50 kW FCS polynomial exactly as specified in Eq. (2) and Table 1. There is no 50-to-60 kW scaling, refitting, or coefficient modification.

## Equations and coefficients

Paper Eq. (2) gives hydrogen consumption rate as a third-order polynomial:

```text
x = P / 100
mdot_H2(P) = a3*x^3 + a2*x^2 + a1*x + a0
```

`P` is FCS power in kW; `x` is power in units of 100 kW; `mdot_H2` is in g/s. Table 1 coefficients are:

```text
a3 =  4.1953
a2 = -2.0662
a1 =  1.6944
a0 =  0.0066
```

In Eq. (1), the paper includes an additional 5% hydrogen purge allowance when deriving hydrogen consumption from fuel-cell efficiency. The fitted Eq. (2) curve used here represents that purge-inclusive consumption. Therefore `mdot_H2` is calculated directly from the polynomial and is **not** multiplied by 1.05 again.

The two saved efficiencies are:

```text
eta_FCS       = 1.05 * P / (120 * mdot_H2)
eta_effective =       P / (120 * mdot_H2)
```

The first follows the paper's Eq. (1) definition and corresponds to the Fig. 2 FCS-efficiency convention. The second omits the 5% factor for compatibility with a consumption-based convention that does not apply a separate purge adjustment. Both are dimensionless fractions in [0, 1] form, not percentages. `LHV = 120 kJ/g`; since kW = kJ/s, the formula returns a dimensionless efficiency when `mdot_H2` is in g/s.

## `fcs_parameters.txt` columns

The file contains exactly four columns, separated by spaces, with no header or comment rows. Values use 17 significant digits and can be loaded with `numpy.loadtxt("fcs_parameters.txt")`.

| Column | Name | Unit | Meaning |
|---|---|---|---|
| 1 | FCS power `P` | kW | Requested sample power, 0 to 50 kW inclusive |
| 2 | Hydrogen consumption `mdot_H2` | g/s | Eq. (2) polynomial result; includes the modeled 5% purge allowance |
| 3 | `eta_FCS` | dimensionless | Eq. (1) / Fig. 2 convention: `1.05*P/(120*mdot_H2)` |
| 4 | `eta_effective` | dimensionless | `P/(120*mdot_H2)`; no additional purge factor |

## Operating constraints and sampling

The paper's Table 3 specifies the normal FCS operating range as 5–50 kW, with a maximum rising power rate of 5 kW/s and a maximum falling power rate of −5 kW/s. These are operating constraints; data below 5 kW are retained here because the requested map covers the polynomial over 0–50 kW. For a simulation time step `dt`, the corresponding rate bounds are `-5*dt <= P[k]-P[k-1] <= 5*dt` kW, in addition to the normal-operation power bounds when the FCS is on.

The sampling interval is 0.1 kW, giving 501 rows. The fitted polynomial is evaluated for every sample from 0 through 50 kW, inclusive.

At `P=0`, the file preserves the polynomial result `mdot_H2 = 0.0066 g/s`, while both efficiencies are set to 0 by the requested convention. This is the polynomial's mathematical value at zero power; it does not represent a physical FCS OFF state. In subsequent simulation, an OFF state should be handled separately with zero hydrogen consumption.

## Numerical check

The sampled `eta_FCS` maximum is 0.596797993431 at 25.8 kW. At 50 kW, hydrogen consumption is 0.861662500000 g/s, `eta_FCS` is 0.507739399127 (about 50.8%), and `eta_effective` is 0.483561332502. This agrees with the paper's Fig. 2 trend and the requested approximate values.
