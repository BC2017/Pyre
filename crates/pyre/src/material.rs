//! Bidirectional scattering distribution functions. All BSDF math runs in
//! the shading-local frame where +Z is the surface normal and `wi`/`wo`
//! are unit vectors above the hemisphere.
//!
//! Milestone 3 added Lambertian. Milestone 4 adds Disney principled
//! (diffuse + GGX specular reflection); transmission/sheen/clearcoat/
//! anisotropic are planned follow-ups.

use glam::{Vec2, Vec3};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct BsdfSample {
    /// Sampled incoming direction in shading-local space. `wi.z > 0` for
    /// reflection above the surface.
    pub wi: Vec3,
    /// BSDF value `f(wo, wi)` (RGB) summed across all lobes.
    pub f: Vec3,
    /// Multi-lobe pdf (`Σ lobe_weight × lobe_pdf`) in solid angle.
    pub pdf: f32,
}

pub trait Bsdf: Send + Sync {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3;
    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32;
    fn sample(&self, wo: Vec3, u: Vec2) -> Option<BsdfSample>;
}

// ---------------------------------------------------------------------------
// Lambertian
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Lambertian {
    pub albedo: Vec3,
}

impl Bsdf for Lambertian {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            Vec3::ZERO
        } else {
            self.albedo / PI
        }
    }

    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            0.0
        } else {
            wi.z / PI
        }
    }

    fn sample(&self, wo: Vec3, u: Vec2) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let wi = cosine_sample_hemisphere(u);
        let pdf = wi.z / PI;
        let f = self.albedo / PI;
        Some(BsdfSample { wi, f, pdf })
    }
}

// ---------------------------------------------------------------------------
// Disney principled BSDF (diffuse + GGX specular reflection)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct DisneyBsdf {
    pub base_color: Vec3,
    /// 0 = pure dielectric, 1 = pure conductor. Linear blend in between.
    pub metallic: f32,
    /// Surface roughness in [0, 1]. Drives `alpha = roughness²` for GGX.
    pub roughness: f32,
    /// Dielectric F0 control: `F0_dielectric = 0.08 × specular`. Default
    /// 0.5 → F0 = 0.04 (≈ glass / plastic, IOR 1.5). Ignored for metals
    /// (their F0 comes from base_color).
    pub specular: f32,
}

impl DisneyBsdf {
    fn alpha(&self) -> f32 {
        // Clamp away from zero so VNDF sampling is numerically stable and
        // glossy lobes don't collapse to a delta (a true delta would need
        // a separate code path).
        (self.roughness * self.roughness).max(1e-3)
    }

    fn f0(&self) -> Vec3 {
        let dielectric = Vec3::splat(0.08 * self.specular);
        dielectric.lerp(self.base_color, self.metallic)
    }

    /// Probability weights for one-sample lobe selection. Driven by the
    /// energy each lobe could plausibly contribute — diffuse weight depends
    /// on (1 - metallic) and base color, specular weight on F0.
    fn lobe_weights(&self) -> (f32, f32) {
        let diff = (1.0 - self.metallic) * luminance(self.base_color);
        let spec = luminance(self.f0()).max(0.04);
        let total = diff + spec;
        if total > 0.0 {
            (diff / total, spec / total)
        } else {
            (1.0, 0.0)
        }
    }

    /// Disney diffuse: Lambert × Fresnel-like retro-reflection factor that
    /// brightens grazing angles when roughness is high (fabric, dry surfaces).
    fn eval_diffuse(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 || self.metallic >= 1.0 {
            return Vec3::ZERO;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() < 1e-10 {
            return Vec3::ZERO;
        }
        let cos_d = wi.dot(h).clamp(0.0, 1.0);
        let fd90 = 0.5 + 2.0 * self.roughness * cos_d * cos_d;
        let fd_v = 1.0 + (fd90 - 1.0) * (1.0 - wo.z).max(0.0).powi(5);
        let fd_l = 1.0 + (fd90 - 1.0) * (1.0 - wi.z).max(0.0).powi(5);
        self.base_color * (1.0 - self.metallic) / PI * fd_v * fd_l
    }

    /// GGX microfacet specular with Schlick Fresnel and height-correlated
    /// Smith shadow-masking.
    fn eval_specular(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return Vec3::ZERO;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() < 1e-10 || h.z <= 0.0 {
            return Vec3::ZERO;
        }
        let alpha = self.alpha();
        let d = ggx_d(h, alpha);
        let f = schlick_fresnel(wi.dot(h).max(0.0), self.f0());
        let g = ggx_g2(wo, wi, alpha);
        d * f * g / (4.0 * wo.z * wi.z)
    }

    fn pdf_diffuse(&self, _wo: Vec3, wi: Vec3) -> f32 {
        if wi.z <= 0.0 {
            0.0
        } else {
            wi.z / PI
        }
    }

    /// VNDF pdf for `wi`: `G1(wo) × D(h) / (4 × |wo·n|)`. The factor of
    /// `|wo·h|` from the visible-normal pdf cancels with the half-vector
    /// reflection Jacobian.
    fn pdf_specular(&self, wo: Vec3, wi: Vec3) -> f32 {
        if wo.z <= 0.0 || wi.z <= 0.0 {
            return 0.0;
        }
        let h = (wo + wi).normalize_or_zero();
        if h.length_squared() < 1e-10 || h.z <= 0.0 {
            return 0.0;
        }
        let alpha = self.alpha();
        let d = ggx_d(h, alpha);
        let g1 = ggx_g1(wo, alpha);
        g1 * d / (4.0 * wo.z)
    }
}

impl Bsdf for DisneyBsdf {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3 {
        self.eval_diffuse(wo, wi) + self.eval_specular(wo, wi)
    }

    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32 {
        let (pd, ps) = self.lobe_weights();
        pd * self.pdf_diffuse(wo, wi) + ps * self.pdf_specular(wo, wi)
    }

    fn sample(&self, wo: Vec3, u: Vec2) -> Option<BsdfSample> {
        if wo.z <= 0.0 {
            return None;
        }
        let (pd, ps) = self.lobe_weights();
        if pd + ps <= 0.0 {
            return None;
        }
        // Reuse u.x both for lobe selection and within the lobe via remap.
        let wi = if u.x < pd {
            let u_remap = Vec2::new((u.x / pd).min(1.0 - 1e-7), u.y);
            cosine_sample_hemisphere(u_remap)
        } else {
            let u_remap = Vec2::new(((u.x - pd) / ps).min(1.0 - 1e-7), u.y);
            let h = sample_ggx_vndf(wo, self.alpha(), u_remap);
            reflect(wo, h)
        };

        if wi.z <= 0.0 {
            return None;
        }

        let pdf = self.pdf(wo, wi);
        if pdf <= 0.0 {
            return None;
        }
        let f = self.eval(wo, wi);
        Some(BsdfSample { wi, f, pdf })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

#[inline]
fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Cosine-weighted hemisphere sampling (Malley's method): sample a disk and
/// project onto the hemisphere.
pub fn cosine_sample_hemisphere(u: Vec2) -> Vec3 {
    let r = u.x.sqrt();
    let phi = 2.0 * PI * u.y;
    let x = r * phi.cos();
    let y = r * phi.sin();
    let z = (1.0 - u.x).max(0.0).sqrt();
    Vec3::new(x, y, z)
}

/// GGX (Trowbridge–Reitz) normal distribution function.
fn ggx_d(h: Vec3, alpha: f32) -> f32 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let denom = h.z * h.z * (a2 - 1.0) + 1.0;
    a2 / (PI * denom * denom)
}

/// Schlick approximation to the Fresnel reflectance.
fn schlick_fresnel(cos_d: f32, f0: Vec3) -> Vec3 {
    let c = cos_d.clamp(0.0, 1.0);
    f0 + (Vec3::ONE - f0) * (1.0 - c).powi(5)
}

fn ggx_lambda(v: Vec3, alpha: f32) -> f32 {
    let cos2 = v.z * v.z;
    if cos2 >= 1.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let tan2 = (1.0 - cos2) / cos2.max(1e-8);
    0.5 * (-1.0 + (1.0 + a2 * tan2).sqrt())
}

fn ggx_g1(v: Vec3, alpha: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(v, alpha))
}

/// Height-correlated Smith masking-shadowing (Heitz 2014).
fn ggx_g2(wo: Vec3, wi: Vec3, alpha: f32) -> f32 {
    1.0 / (1.0 + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha))
}

/// Reflect `v` across `n`: returns `2(v·n)n - v`. Note this is the BSDF-side
/// convention (both vectors point away from the surface).
fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    2.0 * v.dot(n) * n - v
}

/// Heitz 2018 VNDF sampling for isotropic GGX. Returns the sampled
/// microfacet normal in shading-local space.
fn sample_ggx_vndf(wo: Vec3, alpha: f32, u: Vec2) -> Vec3 {
    // Stretch wo into the unit-roughness hemisphere.
    let vh = Vec3::new(alpha * wo.x, alpha * wo.y, wo.z).normalize();

    // Build an orthonormal basis aligned with the stretched view.
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);

    // Sample a point on a projected hemisphere disk.
    let r = u.x.sqrt();
    let phi = 2.0 * PI * u.y;
    let t1c = r * phi.cos();
    let mut t2c = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    t2c = (1.0 - s) * (1.0 - t1c * t1c).max(0.0).sqrt() + s * t2c;

    // Reproject onto the hemisphere.
    let nh = t1c * t1 + t2c * t2 + (1.0 - t1c * t1c - t2c * t2c).max(0.0).sqrt() * vh;

    // Unstretch back to roughness-α space.
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(0.0)).normalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// `sample` must return the same `f` and `pdf` as a direct `eval` / `pdf`
    /// call on the sampled direction — otherwise the integrator's MIS weights
    /// will quietly bias the result.
    #[test]
    fn disney_sample_eval_pdf_consistency() {
        let bsdfs = [
            DisneyBsdf {
                base_color: Vec3::new(0.7, 0.3, 0.2),
                metallic: 0.0,
                roughness: 0.4,
                specular: 0.5,
            },
            DisneyBsdf {
                base_color: Vec3::new(1.0, 0.766, 0.336),
                metallic: 1.0,
                roughness: 0.1,
                specular: 0.5,
            },
            DisneyBsdf {
                base_color: Vec3::splat(0.5),
                metallic: 0.5,
                roughness: 0.7,
                specular: 0.5,
            },
        ];

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEADBEEF);
        for bsdf in &bsdfs {
            for _ in 0..500 {
                // Random wo in the upper hemisphere.
                let cos_t: f32 = rng.random::<f32>() * 0.99 + 0.01;
                let phi: f32 = rng.random::<f32>() * std::f32::consts::TAU;
                let sin_t = (1.0 - cos_t * cos_t).sqrt();
                let wo = Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t);

                let u = Vec2::new(rng.random::<f32>(), rng.random::<f32>());
                let Some(s) = bsdf.sample(wo, u) else {
                    continue;
                };

                let f_check = bsdf.eval(wo, s.wi);
                let p_check = bsdf.pdf(wo, s.wi);
                assert!(
                    (f_check - s.f).length() < 1e-5,
                    "eval mismatch: sample.f = {:?}, eval = {:?}",
                    s.f,
                    f_check
                );
                assert!(
                    (p_check - s.pdf).abs() < 1e-5,
                    "pdf mismatch: sample.pdf = {}, pdf = {}",
                    s.pdf,
                    p_check
                );
                assert!(s.pdf > 0.0);
                assert!(s.wi.z > 0.0);
            }
        }
    }
}
