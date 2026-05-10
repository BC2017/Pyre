//! Bidirectional scattering distribution functions. All BSDF math runs in
//! the shading-local frame where +Z is the surface normal and `wi`/`wo`
//! are unit vectors above the hemisphere.
//!
//! Milestone 3 ships a Lambertian BRDF only. Disney principled BSDF (with
//! GGX specular, transmission, sheen, clearcoat, etc.) lands in milestone 4.

use glam::{Vec2, Vec3};
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy)]
pub struct BsdfSample {
    /// Sampled incoming direction in shading-local space. `wi.z > 0` for
    /// reflection above the surface.
    pub wi: Vec3,
    /// BSDF value `f(wo, wi)` (RGB). For reflective Lambertian: `albedo / π`.
    pub f: Vec3,
    /// Probability density of the sample with respect to solid angle.
    pub pdf: f32,
}

pub trait Bsdf: Send + Sync {
    fn eval(&self, wo: Vec3, wi: Vec3) -> Vec3;
    fn pdf(&self, wo: Vec3, wi: Vec3) -> f32;
    fn sample(&self, wo: Vec3, u: Vec2) -> Option<BsdfSample>;
}

/// Pure Lambertian (perfectly diffuse) BRDF.
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
