##KN1DJL vs KN1DPY
This version of KN1D is maintained separately than the python version found here (https://github.com/W-M-plasma-group/KN1DPy). A set of common test cases are being developed to ensure general parity between the two.

## KN1D

KN1D is a 1D-space, 2D-velocity kinetic neutrals code developed by B. LaBombard (MIT PSFC). KN1Djl is a Julia translation of the python version.
This translation was written and maintained by Sean Lyons at UCSD (splyons@ucsd.edu) with help from Jamie Dunsmore at MIT (jduns@mit.edu).

A comprehensive description of the algorithm can be found in this PSFC report (LaBombard, 2001).

Please note this translation is new albeit complete, so if any bugs are found please email "splyons@ucsd.edu."

## Inputs

The main entry point is `kn1d()` in `KN1DPy/kn1d.py`. All array inputs should be 1D numpy arrays of length `nx`, defined on the radial coordinate grid `x`.

| Parameter  | Type            | Units         | Description |
|------------|-----------------|---------------|-------------|
| `x`        | ndarray (nx)    | m             | x-coordinate |
| `xlimiter` | float           | m             | Limiter position |
| `xsep`     | float           | m             | Separatrix position |
| `GaugeH2`  | float           | mTorr         | Molecular neutral pressure at the wall |
| `mu`       | float           | —             | Ion mass: 1 = hydrogen, 2 = deuterium |
| `Ti`       | ndarray (nx)    | eV            | Ion temperature profile |
| `Te`       | ndarray (nx)    | eV            | Electron temperature profile |
| `n`        | ndarray (nx)    | m⁻³           | Electron density profile |
| `vxi`      | ndarray (nx)    | m s⁻¹         | Plasma flow velocity (negative = towards wall). Generally set this to 0 |
| `LC`       | ndarray (nx)    | m             | Connection length to nearest limiter along field lines (0 = infinity) |
| `PipeDia`  | ndarray (nx)    | m             | Effective diameter of the pressure gauge pipe for side-wall collisions (0 = disabled, which is the most common setting) |

---

## Outputs

When `File` is specified, `kn1d()` writes four `.npz` files and a copy of the active `config.toml` to the output directory. The `.npz` files can be loaded with `numpy.load()`.

| File | Description |
|------|-------------|
| `KN1D_H.npz` | Atomic hydrogen results. The core output is `fH`, the 2D velocity distribution function (vr × vx) on the atomic spatial grid `xH`. All atomic quantities — density `nH`, particle flux `GammaxH`, temperature `TH`, ionization source `Sion`, emissivities, and higher-order moments — are derived from this distribution function. |
| `KN1D_H2.npz` | Molecular hydrogen results. The core output is `fH2`, the 2D velocity distribution function (vr × vx) on the molecular spatial grid `xH2`. Derived quantities include `nH2`, `GammaxH2`, `TH2`, and the atomic and ion source terms `SH` and `SP`. |
| `KN1D_input.npz` | The input profiles (`Ti`, `Te`, `n`, etc.) interpolated onto both the atomic and molecular spatial grids, together with the velocity grid arrays (`vrA`, `vxA`, `vrM`, `vxM`). Useful for plotting inputs and outputs on the same axes. |
| `KN1D_mesh.npz` | The raw velocity and spatial grid parameters used internally. |
| `config.toml` | A copy of the configuration used for this run, so the outputs are fully self-documenting. |

---

## Configuration

As well as the inputs, the settings for each run (e.g choice of atomic rate coefficients and mesh sizes) can be controlled via `config.toml` in the root directory (or a custom path passed via `config_path`). The settings that can be changed in the `config.toml` file are as follows...

### `kinetic_h` and `kinetic_h2`

| Key              | Description |
|------------------|-------------|
| `mesh_size`      | Number of velocity grid points. Likely needs to be increased from default for cases with > 500 eV pedestals. |
| `grid_fctr`      | Scales the physics-based maximum spatial grid spacing. Smaller values give a finer mesh. Default `0.3` (matching the IDL default). Lower values mean finer resolution. |
| `extra_energy_bins_eV` | Additional velocity grid points at the specified energies (eV). `kinetic_h` has no hardcoded energy bins, but `kinetic_h2` already has hardcoded energy bins at `0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0`, so any `kinetic_h2` inputs here will be in addition to these. Default `[]`. |
| `ion_rate`       | Ionization rate method: `"adas"` (recommended), `"jh"` (Johnson–Hinnov), `"collrad"`, or `"janev"`. (`kinetic_h` only) |

### `collisions`

Each flag enables or disables a specific collision channel:

| Key         | Description |
|-------------|-------------|
| `H2_H2_EL`  | H₂ → H₂ elastic self-collisions |
| `H2_P_EL`   | H₂ → H⁺ elastic collisions |
| `H2_H_EL`   | H₂ ↔ H elastic collisions |
| `H2_P_CX`   | H₂ → H₂⁺ charge exchange |
| `H_H_EL`    | H → H elastic self-collisions |
| `H_P_CX`    | H → H⁺ charge exchange |
| `H_P_EL`    | H → H⁺ elastic collisions |
| `SIMPLE_CX` | Use simplified CX collisions (neutrals born with ion distribution) |

---

## kn1d_lite

`kn1d_lite` is a simplified version of `kn1d` that is designed to be run on closed field lines (inside the separatrix).
It ignores molecules, and simply calculates the atomic neutral density for specified plasma profiles on the closed field lines. See [docs/kn1d_lite.md](docs/kn1d_lite.md) for a full description and instructions on how to run. Given the limited information on edge conditions available in the current version of FUSE kn1d_lite is what is used there.

