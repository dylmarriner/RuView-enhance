# Phase-coherent angle-of-arrival (AoA)

Module: `wifi_densepose_signal::aoa` (`v2/crates/wifi-densepose-signal/src/aoa/`).

> **Evidence status: SYNTHETIC only.** Every accuracy figure below comes from
> plane-wave simulations in the unit tests. There is **no hardware evidence**:
> the current deployment uses single-antenna ESP32-S3 boards, which have one RF
> chain and cannot measure an angle of arrival. A successful build or test run
> is not evidence that the algorithm works on real silicon. See
> "Hardware needed" and "Limitations".

## What it does

Takes multi-antenna CSI `[antenna, subcarrier, packet]` from a **phase-coherent**
receiver and produces bearings:

1. Phase calibration (per-antenna complex offsets from a reference capture) and
   per-packet common-phase/slope removal.
2. Spatial covariance, forward-backward averaging, spatial smoothing.
3. Bartlett, MVDR/Capon and MUSIC spatial spectra (1-D azimuth for linear
   arrays, 2-D azimuth/elevation for planar arrays), MDL/AIC source counting,
   peak picking with parabolic sub-grid refinement.
4. Optional SpotFi-style joint AoA + relative ToF (`spotfi_music`).
5. Fusion of bearings from several arrays at known poses into a 2-D position
   (`locate_2d`).

Quick use:

```rust
use wifi_densepose_signal::aoa::*;

let geom = ArrayGeometry::ura_half_wave(4, 2, 2.437e9)?;      // ESPARGOS-like 2x4
let snaps = snapshots_from_iq_frames(8, &frames)?;             // antenna-major I/Q
let snaps = Calibration::estimate(&ref_capture, Reference::Common, true)?.apply(&snaps)?;
let res = estimate_aoa(&snaps, &geom, &AoaConfig::default())?; // MUSIC + MDL
for p in &res.peaks { println!("az {:.1} el {:.1}", p.azimuth.to_degrees(), p.elevation.to_degrees()); }
```

## Maths

Conventions: azimuth is CCW from the array +x axis, elevation from the xy plane.
A far-field plane wave from unit direction `u` gives phase `exp(+j k p·u)` at
antenna position `p`, `k = 2π/λ`; the steering vector is
`a(θ,φ)_m = exp(j k p_m·u(θ,φ))`. Snapshot model `x = A s + n`.

* **Covariance**: `R = (1/N) Σ x xᴴ` over all `(subcarrier, packet)` columns
  (`N = K·P ≥ M` required). Narrowband approximation: steering is evaluated at
  the channel centre frequency; `ArrayGeometry::steering_at` gives the exact
  per-subcarrier version. Across a 20 MHz channel at 2.4 GHz the wavelength
  varies by under 1 %.
* **Bartlett**: `P(θ) = aᴴ R a / aᴴ a` (robust, low resolution).
* **MVDR/Capon**: `P(θ) = 1 / aᴴ (R + δI)⁻¹ a`, `δ = loading · tr(R)/M`.
* **MUSIC**: eigendecompose `R`; with `d` sources and noise subspace `E_n`,
  `P(θ) = aᴴa / aᴴ E_n E_nᴴ a`. Uses nalgebra's complex Hermitian
  `SymmetricEigen`.
* **Forward-backward**: `R_fb = ½ (R + J R* J)`, `J` = point-reflection
  permutation of the array about its centroid (requires a centro-symmetric
  array; otherwise an error). Decorrelates two coherent paths.
* **Spatial smoothing**: average of `R` over shifted `sx × sy` sub-arrays of a
  ULA/URA grid; restores rank for coherent multipath at the cost of aperture
  (`M − sub + 1` shifts resolve that many coherent sources).
* **Source count**: Wax-Kailath MDL (default) / AIC on the eigenvalues. Adjacent
  subcarriers are correlated, so `N` is optimistic; override with a fixed
  `n_sources` when known.
* **Sub-grid**: parabola through `ln P` at the peak and its two neighbours.
* **SpotFi-style joint AoA/ToF**: a window of `Ls` subcarriers over `M` antennas
  gives an `M·Ls` vector with steering `a(θ) ⊗ exp(−j 2π Δf l τ)`; MUSIC over
  `(θ, τ)`. Absolute ToF is unobservable (unknown sampling-time offset); only
  path delay differences are meaningful.
* **Fusion**: each bearing is a ray; minimise `Σ w_i d_i²` (perpendicular
  distances) in closed form, iteratively re-weighting with `σ_i·range_i`.
  Returns position, 2×2 covariance, residual RMS and per-ray ranges.

## Calibration procedure

Phase-synchronous chains still have fixed per-chain complex offsets.

1. **Shared reference** (ESPARGOS-style): feed one reference signal to every
   antenna through equal-length paths, capture ≥ 8 packets, call
   `Calibration::estimate(&cap, Reference::Common, per_subcarrier)`.
   The dominant covariance eigenvector is the offset vector, referenced to
   antenna 0 and normalised to unit mean gain. `per_subcarrier = true` also
   models frequency-dependent cable delay (needs ≥ `n_antennas` packets).
2. **Known bearing**: transmit from a surveyed direction (far field, line of
   sight, no reflectors nearby) and use `Reference::Bearing { .. }`.
3. Apply to every later capture with `Calibration::apply`. Recalibrate after
   any power cycle or re-clock that can change the PLL phase relationship, and
   check drift against temperature.
4. `remove_common_phase` removes the per-packet slope/constant phase identically
   on all antennas (CFO/SFO/STO residue) without touching inter-antenna phase.

## Hardware needed

| Option | Notes |
|---|---|
| ESPARGOS-like phase-synchronous array | 2×4 ESP32 array sharing a reference clock and a reference signal for calibration (concept only; the pyespargos code is LGPL-3.0 and was not used). |
| Espressif esp-crab (esp-csi, Apache-2.0) | RF phase synchronisation of two ESP32-C5 chips sharing a clock; 2 chains per pair. |
| Intel 5300 / AX210 etc. (3 chains) | Multi-antenna NICs via other CSI tools; per-NIC chain phase offsets must be calibrated, 3 antennas give limited resolution. |

A single ESP32-S3 (current RuView nodes) has one RF chain; several unsynchronised
S3 boards are **not** a phase-coherent array because each has an independent
oscillator.

Raw I/Q layout expected by `snapshots_from_iq_frames`: antenna-major (all
subcarriers of antenna 0, then antenna 1, …), which is what the RuView ESP32
parser produces.

## SYNTHETIC accuracy (unit tests)

All values: simulated plane waves, complex Gaussian source symbols and noise,
8-element λ/2 ULA unless noted, 2.437 GHz, 16 subcarriers × 8 packets = 128
snapshots, fixed RNG seeds. SNR is per-antenna, per-source. Run
`cargo test -p wifi-densepose-signal aoa -- --nocapture` to print the exact
errors; the asserted bounds are deliberately looser than the achieved values. These errors are tiny because the model is noise-only: no coupling, calibration residue or real multipath.

| Scenario | SNR | Result (SYNTHETIC) |
|---|---|---|
| 1 source @70°, MUSIC | 10 dB | MUSIC 0.03°, Bartlett 0.08°, MVDR 0.07° (asserted < 0.5° / 1° / 0.75°) |
| 2 uncorrelated sources 55°/85°, MDL count + MUSIC | 15 dB | count = 2, errors 0.01° and 0.003° (asserted < 1°) |
| 2 fully coherent paths 50°/100°, FB + smoothing (5-elem subarrays) | 20 dB | errors < 0.02° (asserted < 2°); plain MUSIC worst error 14.8° |
| 4×4 URA, az 40° / el 30° | 15 dB | az 0.02°, el 0.01° (asserted < 1° / 1.5°) |
| 2×4 URA (ESPARGOS-like) same source | 15 dB | az 0.02°, el 0.00° (asserted < 1.5° / 3°) |
| Random per-antenna offsets recovered (shared reference) | 25 dB | phase 0.014 rad, gain 0.008; AoA error 27° uncalibrated -> 0.01° calibrated |
| Two ULAs 6 m apart, source at (2, 3) m, 1° bearing σ | 15 dB | position error 0.008 m (asserted < 0.1 m) |
| SpotFi 3-ant, 30 subcarriers, 2 paths | 25 dB | AoA < 0.02°, ToF < 0.1 ns (asserted < 3° / 15 ns) |
| RMS over 20 trials, 1 source | 0 dB / 10 dB | 0.40° / 0.05° (asserted < 3° / < 1°) |

## Limitations

* **No hardware validation.** Real arrays add mutual coupling, element pattern
  differences, calibration drift, phase noise and multipath that these tests do
  not model. Real accuracy will be worse; measure it and tag it `MEASURED`.
* Narrowband steering: very wide channels (80/160 MHz) or large apertures need
  per-subcarrier steering (squint). Not done in the covariance path.
* Linear arrays see only `cos(azimuth)`: front/back ambiguity, poor resolution
  near end-fire. Planar 2×N arrays resolve elevation weakly (two rows).
* Planar arrays in the xy plane cannot distinguish above from below; default
  elevation scan is `[0°, 90°]`.
* MDL assumes independent snapshots; subcarrier snapshots are correlated.
* Coherent-source handling needs a grid array and costs aperture.
* Resolution is bounded by aperture (about `1/(M·cos θ)` rad); a 4-element array
  cannot resolve closely spaced people.
* WiFi AoA is not camera-grade localisation; indoors, reflections routinely
  dominate the line-of-sight path.
* Absolute ToF/range is not observable from one array.
