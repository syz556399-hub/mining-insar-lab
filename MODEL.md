# Model specification · engine working-face-integral-2.1

This document specifies the implemented model rather than a calibrated prediction method. All assumptions below are part of version 2.1.0. Code is independently written; Gaussian influence kernels, radar phase relations and complex Gaussian statistics are established ideas. [Scientific references and boundaries](docs/REFERENCES.md).

## 1. Coordinates, geometry and units

The square scene has N×N pixel centers, side S m, spacing p=S/N. Columns increase eastward E; rows increase southward, so North N decreases with row. No CRS, georeferencing or radar-native range/azimuth grid is defined. Faces occupy horizontal seams under a locally flat mechanical surface.

For panel center (Ec,Nc), length L, width B and advance bearing θ measured counterclockwise from East:

u=(E−Ec)cosθ+(N−Nc)sinθ,
v=−(E−Ec)sinθ+(N−Nc)cosθ.

`radar_azimuth_deg` uses a different convention: North=0°, East=90°, ground toward satellite. Every exported length/displacement is m, time day, angle phase rad; GUI mm/degree fields are converted before simulation.

## 2. Rectangular influence and segmented time

Influence radius r=H/tanβ, normalized one-dimensional kernel g(x)=exp(−πx²/r²)/r. The analytical interval integral is

I(x;a,b)=½[erf(√π(x−a)/r)−erf(√π(x−b)/r)].

Divide [−L/2,L/2] into K equal strips [aj,bj]. Each strip activates at its center-passage time tj=tstart+[(aj+bj)/2+L/2]/speed. Response Fj(t)=1−exp(−max(0,t−tj)/τ).

For seam thickness m and subsidence factor q:

W(t,E,N)=mq Σj I(u;aj,bj) I(v;−B/2,B/2) Fj(t).

W is downward-positive. Compute W at both epochs, then ΔD=W(t1)−W(t0). Disabled faces contribute zero; multiple face displacements add before phase calculation. Spatial integration is analytical; strip-center activation is a time discretization. It can miss short intervals between strip activation times, especially with few strips, narrow influence radius or rapid response. Assess K convergence for each intended regime. The GUI filled panel is continuous advance geometry, not the exact activation step pattern.

The model omits seam inclination, overburden mechanics, nonlinear panel interactions, asymmetric boundary corrections and fitting to mine observations. A more realistic-looking result is not evidence of improved predictive accuracy.

## 3. Horizontal approximation and radar signs

For each face ΔE=b r ∂ΔD/∂E and ΔN=b r ∂ΔD/∂N; gradients use signed pixel spacing so North is handled correctly. This is an empirical slope approximation with dimensionless b. It is not a solution of elastic/rock mechanics.

For incidence i from vertical and radar azimuth a:

ΔLOS=ΔD cos(i)−[ΔE sin(a)+ΔN cos(a)]sin(i),
φd=−4πΔLOS/λ.

Positive LOS is increasing ground-satellite distance. Interferometric product is S1 conj(S2); a phase sign convention must be converted consistently before comparison with another processing system.

## 4. Synthetic environment and phase components

Synthetic height comes from filtering real white noise in the Fourier domain by (k²+k0²)^(−β/4), inverse FFT, then scaling the relief to the requested min/max range above an arbitrary 300 m base. Its power spectrum is asymptotically k^(−β); no claim of measured terrain or Kolmogorov atmospheric calibration is made. DEM error uses a separate normalized random field with requested standard deviation. Terrain, water and error use separate random streams.

River and ellipse-lake masks are geometric constructions. They do not follow a DEM-derived watershed, fill a constant-elevation basin or reproduce the supplied BMP. Their geometry and power contrast are simplified assumptions.

Small-baseline coefficient T=−4π B⊥/(λ R sin(i)). Full terrain phase is T(h−mean(h)); differential mode uses Tδh, with δh defined by this residual sign convention. Both vanish at zero baseline; differential terrain phase vanishes at zero DEM error. Nonzero baseline at incidence <1° is rejected to avoid singular geometry. This is a local constant-geometry approximation, not slant-range resampling or topographic shadow/layover modeling.

Turbulent-like background consists of 24 random plane waves normalized to zero mean and the requested equivalent-LOS standard deviation. It has no prescribed measured turbulence spectrum. Stratified atmosphere adds (h−mean(h))/100 × height_atmosphere_mm/1000 m. The orbit term is 2π×orbit_cycles×E/S. Reference phase:

φref=φd−4π A/λ+φorbit+φtopography,applied.

Changing DEM affects observation-phase terms and the empirical coherence map, but does not change the underlying mechanical subsidence/face depth calculation.

## 5. Complex observation and sample coherence

Land γ=γland exp(−Δt/Tc) exp(−cslope|∇h|); Tc=0 disables time attenuation. Water replaces this value with γwater. A Gaussian-shaped disturbance may reduce γ further. These empirical laws have not been fitted to actual landcover/time/baseline data. Model normalized power P=1 on land and 0.15 on water, ignoring thermal receiver noise and frequency-dependent scattering.

For independent standard circular complex Gaussian z,n at each pixel and realization:

S1=√P z,
S2=√P[γz+√(1−γ²)n]exp(−iφref).

Thus E[S1 conj(S2)]=Pγ exp(iφref). Accumulate M independent realizations' cross-products and powers. Apply the same odd square sliding mean (reflected boundaries, unchanged output dimensions) to each. The observed phase is arg(mean cross-product); estimated coherence is |mean cross-product|/√(mean power1×mean power2), clipped only for floating-point roundoff.

Finite-sample coherence is positively biased, particularly at low γ. A varying phase within a window lowers magnitude even at model γ=1. Overlapping windows create correlated outputs. Independent ensembles and window area are not a guarantee of their product being a physical number of independent SAR looks. A single realization at window 1 yields estimate 1 regardless of true γ; the application warns. The unsmoothed reference and labels remain pixel truth, so spatial smoothing can reduce sharp input/label agreement near boundaries.

Legacy `complex_observation=False` uses von Mises circular phase errors and complex resultant averaging. Its exported `estimated_coherence` is a resultant length, **not** SAR sample coherence; this is marked by `estimated_coherence_kind`. Legacy gamma maps do not drive that observation model. Legacy mode is for controlled comparison, not the UI default.

## 6. Labels, validity and precision

mask=[float32(ΔD) ≥ target_threshold_mm/1000]. Each face mask thresholds its own increment. The total mask can differ from the union of face masks because subthreshold increments can add. Physical labels are defined before observation disturbances; water does not force underlying displacement to zero.

valid_mask=[water_mask=0 AND float32(model_coherence) ≥ valid_coherence_threshold]. This is a declared synthetic supervision rule, not measured data-quality truth. It does not buffer spatial-window support around water edges or mask real radar shadows. Use this mask explicitly in loss/evaluation.

Computations primarily use float64/complex128; arrays are exported float32 or uint8 (binary masks). Stored cumulative epochs can round differently from their stored difference; consistency tests therefore use tolerances. Wrapped phase theoretically lies within (−π,π], with float32 endpoint rounding. RGB phase visualization uses cyclic cosine channels and is not numerical truth.

## 7. Provenance and limits of validation

Exports carry version, normalized parameters/checksum, group identity, array schema, versions and scope. [Reproducibility](docs/REPRODUCIBILITY.md) and [sampling](docs/SAMPLING.md) explain exact replay and data splitting. Old exports are not silently replayed by a new stochastic engine.

Analytical/statistical tests establish implementation properties. No leveling/GNSS comparison, fitted mine-specific parameters, real DEM alignment, physical SLC simulation or synthetic-to-real detection result has been established. The supplied BMP lacks a verified color-to-phase mapping, pixel spacing and geographic metadata; optional local previews provide qualitative comparison only. Public code bundles omit it.
