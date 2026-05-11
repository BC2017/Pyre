//! Inverse-CDF sampling helpers. PBRT chapter 13 covers the math; the
//! flavour here is straight piecewise-constant 1D and 2D distributions
//! used by the HDRI environment light to importance-sample bright pixels.

use glam::Vec2;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

/// Concentric mapping from a square sample `[0,1)^2` to the unit disk
/// (Shirley & Chiu 1997). Equal-area, lower distortion than the naive
/// polar map — keeps stratification properties of the input sampler.
/// Used by `ThinLensCamera` for aperture sampling.
pub fn concentric_disk(u: Vec2) -> Vec2 {
    let offset = 2.0 * u - Vec2::ONE;
    if offset == Vec2::ZERO {
        return Vec2::ZERO;
    }
    let (r, theta) = if offset.x.abs() > offset.y.abs() {
        (offset.x, FRAC_PI_4 * (offset.y / offset.x))
    } else {
        (offset.y, FRAC_PI_2 - FRAC_PI_4 * (offset.x / offset.y))
    };
    Vec2::new(r * theta.cos(), r * theta.sin())
}

/// Piecewise-constant 1D distribution over `[0, 1)`. Built from an array
/// of non-negative weights; sampling returns a continuous index in
/// `[0, n)` (i.e. weight bin + intra-bin offset) plus the pdf.
#[derive(Debug, Clone)]
pub struct Distribution1D {
    /// `cdf[i]` = sum of weights `[0..i)` divided by total. `cdf[0] = 0`,
    /// `cdf[n] = 1` (provided total > 0).
    cdf: Vec<f32>,
    /// Sum of input weights — used to recover the pdf of any bin from
    /// the original weights.
    integral: f32,
    n: usize,
}

impl Distribution1D {
    pub fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        let mut acc = 0.0_f32;
        for &w in weights {
            acc += w.max(0.0);
            cdf.push(acc);
        }
        let integral = acc;
        if integral > 0.0 {
            for c in cdf.iter_mut() {
                *c /= integral;
            }
        } else {
            // Degenerate — fall back to uniform so sampling stays defined.
            for (i, c) in cdf.iter_mut().enumerate() {
                *c = i as f32 / n as f32;
            }
        }
        Self { cdf, integral, n }
    }

    pub fn integral(&self) -> f32 {
        self.integral
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Continuous inverse-CDF sample. Returns `(t, pdf, bin)` where:
    /// - `t ∈ [0, 1)` is the continuous sample location,
    /// - `pdf` is the density of `t` (with respect to `dt`),
    /// - `bin` is the index of the chosen weight (useful for 2D nesting).
    pub fn sample_continuous(&self, u: f32) -> (f32, f32, usize) {
        // Binary-search upper-bound: smallest `i` with `cdf[i] > u`.
        let mut lo = 1;
        let mut hi = self.cdf.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cdf[mid] <= u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let bin = (lo - 1).min(self.n - 1);
        let cdf_lo = self.cdf[bin];
        let cdf_hi = self.cdf[bin + 1];
        let bin_size = (cdf_hi - cdf_lo).max(1e-20);
        // Linear remap of `u` within the bin.
        let du = (u - cdf_lo) / bin_size;
        let t = (bin as f32 + du) / self.n as f32;
        // pdf with respect to `t`: a unit-length integration domain.
        let pdf = bin_size * self.n as f32;
        (t, pdf, bin)
    }

    /// pdf at continuous location `t` ∈ `[0, 1)`.
    pub fn pdf_at(&self, t: f32) -> f32 {
        if t < 0.0 || t >= 1.0 || self.n == 0 {
            return 0.0;
        }
        let bin = ((t * self.n as f32) as usize).min(self.n - 1);
        let cdf_lo = self.cdf[bin];
        let cdf_hi = self.cdf[bin + 1];
        (cdf_hi - cdf_lo) * self.n as f32
    }
}

/// Piecewise-constant 2D distribution stored as a marginal-over-rows plus
/// per-row conditional. Layout matches the equirectangular HDRI use case:
/// `weights[y * width + x]`, with `y` (the marginal axis) running fastest
/// across rows and `x` (the conditional axis) being the columns.
pub struct Distribution2D {
    width: usize,
    height: usize,
    /// `conditionals[y]` is a 1D distribution over the row at `y`.
    conditionals: Vec<Distribution1D>,
    /// Marginal distribution over rows, weighted by row integrals. Sampling
    /// this picks a row, then we sample within the conditional.
    marginal: Distribution1D,
}

impl Distribution2D {
    pub fn new(weights: &[f32], width: usize, height: usize) -> Self {
        assert_eq!(weights.len(), width * height);
        let mut conditionals = Vec::with_capacity(height);
        let mut row_integrals = Vec::with_capacity(height);
        for y in 0..height {
            let row = &weights[y * width..(y + 1) * width];
            let dist = Distribution1D::new(row);
            row_integrals.push(dist.integral());
            conditionals.push(dist);
        }
        let marginal = Distribution1D::new(&row_integrals);
        Self {
            width,
            height,
            conditionals,
            marginal,
        }
    }

    /// Sample a `(u, v)` in `[0,1)^2`. Returns `((u, v), pdf)` with `pdf`
    /// in the joint `(u, v)` domain.
    pub fn sample_continuous(&self, sample: Vec2) -> (Vec2, f32) {
        let (v, pdf_v, row) = self.marginal.sample_continuous(sample.y);
        let (u, pdf_u, _col) = self.conditionals[row].sample_continuous(sample.x);
        (Vec2::new(u, v), pdf_u * pdf_v)
    }

    pub fn pdf_at(&self, uv: Vec2) -> f32 {
        if uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0 {
            return 0.0;
        }
        let row = ((uv.y * self.height as f32) as usize).min(self.height - 1);
        self.conditionals[row].pdf_at(uv.x) * self.marginal.pdf_at(uv.y)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution1d_uniform_samples_uniformly() {
        let d = Distribution1D::new(&[1.0; 4]);
        let (t, pdf, bin) = d.sample_continuous(0.0);
        assert!((t - 0.0).abs() < 1e-6);
        assert!((pdf - 1.0).abs() < 1e-6);
        assert_eq!(bin, 0);

        let (t, pdf, bin) = d.sample_continuous(0.999_999);
        assert!(t < 1.0 && t >= 0.75);
        assert!((pdf - 1.0).abs() < 1e-3);
        assert_eq!(bin, 3);
    }

    #[test]
    fn distribution1d_concentrates_mass() {
        // 90% of mass in bin 1; sampling u=0.5 should land there.
        let d = Distribution1D::new(&[0.05, 0.9, 0.05]);
        let (t, pdf, bin) = d.sample_continuous(0.5);
        assert_eq!(bin, 1);
        assert!(t >= 1.0 / 3.0 && t < 2.0 / 3.0);
        // pdf in bin 1 is 0.9 * 3 = 2.7.
        assert!((pdf - 2.7).abs() < 1e-3);
    }

    #[test]
    fn distribution2d_pdf_normalizes() {
        // Riemann sum of pdf over the unit square should be ~1.
        let weights: Vec<f32> = (0..16).map(|i| (i as f32) + 1.0).collect();
        let d = Distribution2D::new(&weights, 4, 4);
        let mut acc = 0.0;
        let n = 32usize;
        for y in 0..n {
            for x in 0..n {
                let u = (x as f32 + 0.5) / n as f32;
                let v = (y as f32 + 0.5) / n as f32;
                acc += d.pdf_at(Vec2::new(u, v));
            }
        }
        let cell = 1.0 / (n * n) as f32;
        let total = acc * cell;
        assert!((total - 1.0).abs() < 1e-2, "total={total}");
    }
}
