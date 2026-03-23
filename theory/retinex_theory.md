# Retinex Theory

## 1. Retinex Concept

The **Retinex** model (a portmanteau of *retina* and *cortex*) was introduced by
Edwin Land in 1977. It decomposes an observed image **I(x, y)** into two
components:

| Component        | Symbol  | Meaning                                           |
| ---------------- | ------- | ------------------------------------------------- |
| **Reflectance**  | R(x, y) | Intrinsic surface colour, independent of lighting |
| **Illumination** | L(x, y) | Scene lighting, which varies slowly across space  |

The relationship is multiplicative:

```
I(x, y) = R(x, y) × L(x, y)
```

The goal is to **estimate and remove** the illumination component so that only
the reflectance (true appearance) remains.

### 1.1 Single-Scale Retinex (SSR)

SSR estimates the illumination by applying a Gaussian blur to the image. Working
in log-space converts the multiplication into subtraction:

```
R_ssr(x, y) = log( I(x, y) ) − log( G_σ(x, y) ∗ I(x, y) )
```

where **G_σ** is a Gaussian kernel with standard deviation **σ** (the *scale*).

- **Small σ** → preserves fine detail but poor dynamic-range compression.
- **Large σ** → strong dynamic-range compression but may lose local contrast.

### 1.2 Multi-Scale Retinex (MSR)

MSR addresses SSR's single-scale trade-off by averaging the results of SSR at
several scales:

```
R_msr(x, y) = (1/N) Σ [ log( I(x, y) ) − log( G_σₙ ∗ I(x, y) ) ]
```

Typical scale triplet: **σ ∈ {15, 80, 250}** (Jobson et al., 1997).

This yields a balanced result with both good local contrast *and* good
dynamic-range compression.

---

## 2. Human Perception

Human vision exhibits **lightness constancy** — the ability to perceive a
surface's colour as roughly the same under widely varying illumination. For
example, a white sheet of paper looks "white" whether viewed indoors under
tungsten light or outdoors in daylight, even though the actual wavelengths
reaching the eye differ dramatically.

Retinex theory models this by assuming the visual system estimates the
illumination independently and "divides it out", leaving the perceived
reflectance. This is why Retinex-enhanced images often look more natural to
human observers: the algorithm mimics the perceptual normalisation our eyes
already perform.

---

## 3. Strengths and Limitations

### Strengths

- **Enhanced detail in shadows**: Reveals information hidden in under-exposed
  regions.
- **Dynamic-range compression**: Reduces the extreme contrast between bright
  and dark areas, similar to HDR tone-mapping.
- **More natural colour**: Because it models human perception, enhanced images
  tend to look perceptually pleasing rather than artificially boosted.

### Limitations

- **Halo artefacts**: Near strong edges the Gaussian blur can bleed across
  boundaries, producing visible halos.
- **Colour distortion**: Without additional colour restoration (e.g. MSRCR),
  colours can shift or desaturate.
- **Computational cost**: Large Gaussian kernels (σ = 250) require significant
  convolution work, though FFT-based implementations mitigate this.
- **Parameter sensitivity**: The choice of σ values and the number of scales
  affects output quality and must be tuned per application.

---

## References

- Land, E. H. (1977). *The Retinex Theory of Color Vision*. Scientific American.
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). *A Multiscale Retinex for
  Bridging the Gap Between Color Images and the Human Observation of Scenes*.
  IEEE Transactions on Image Processing.
