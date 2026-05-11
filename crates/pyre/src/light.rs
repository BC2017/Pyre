//! Light sources. The path tracer uses each light for two things:
//! - **direct sampling** (next-event estimation): pick a point on the light
//!   from a surface hit and weight by `Li / pdf` (with MIS against BSDF
//!   sampling);
//! - **incidental hits**: when a BSDF-sampled ray happens to hit the light
//!   geometry, the integrator pulls `Le` directly so the path tracer remains
//!   unbiased even without NEE.

use crate::distribution::Distribution2D;
use crate::math::Ray;
use glam::{Vec2, Vec3};
use std::f32::consts::{FRAC_1_PI, PI, TAU};

#[derive(Debug, Clone, Copy)]
pub struct LightSample {
    /// Sampled point on the light surface in world space.
    pub position: Vec3,
    /// Unit direction from the shading point toward `position`.
    pub wi: Vec3,
    /// Distance from shading point to `position` (for visibility).
    pub distance: f32,
    /// Emitted radiance from the sampled point in direction `-wi`.
    pub li: Vec3,
    /// Probability density of the sample with respect to solid angle at
    /// the shading point.
    pub pdf: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LightHit {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
}

pub trait Light: Send + Sync {
    fn sample(&self, from: Vec3, u: Vec2) -> Option<LightSample>;
    /// PDF (solid-angle, at `from`) of having sampled the direction `wi`.
    fn pdf(&self, from: Vec3, wi: Vec3) -> f32;
    fn le(&self, position: Vec3, normal: Vec3, wo: Vec3) -> Vec3;
    fn intersect(&self, ray: &Ray) -> Option<LightHit>;
}

/// One-sided emissive parallelogram. Emits along `+normal` (where
/// `normal = (edge_u × edge_v).normalize()`); the back face is dark.
#[derive(Debug, Clone, Copy)]
pub struct DiffuseAreaQuadLight {
    p0: Vec3,
    edge_u: Vec3,
    edge_v: Vec3,
    /// `edge_u × edge_v` — magnitude is the area, direction is the (un-normalized) normal.
    cross: Vec3,
    /// `1 / |cross|²` — used to invert the (edge_u, edge_v) parameterization.
    inv_cross_len_sq: f32,
    normal: Vec3,
    area: f32,
    emission: Vec3,
}

impl DiffuseAreaQuadLight {
    pub fn new(p0: Vec3, edge_u: Vec3, edge_v: Vec3, emission: Vec3) -> Self {
        let cross = edge_u.cross(edge_v);
        let area = cross.length();
        let inv_cross_len_sq = 1.0 / cross.length_squared();
        let normal = cross.normalize();
        Self {
            p0,
            edge_u,
            edge_v,
            cross,
            inv_cross_len_sq,
            normal,
            area,
            emission,
        }
    }

    pub fn area(&self) -> f32 {
        self.area
    }

    pub fn normal(&self) -> Vec3 {
        self.normal
    }

    fn intersect_plane(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Option<(f32, Vec3)> {
        let denom = dir.dot(self.normal);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = (self.p0 - origin).dot(self.normal) / denom;
        if t < t_min || t > t_max {
            return None;
        }
        let p = origin + t * dir;
        let d = p - self.p0;
        // Solve d = α·edge_u + β·edge_v for (α, β) using cross-product identities.
        let alpha = d.cross(self.edge_v).dot(self.cross) * self.inv_cross_len_sq;
        if !(0.0..=1.0).contains(&alpha) {
            return None;
        }
        let beta = self.edge_u.cross(d).dot(self.cross) * self.inv_cross_len_sq;
        if !(0.0..=1.0).contains(&beta) {
            return None;
        }
        Some((t, p))
    }
}

impl Light for DiffuseAreaQuadLight {
    fn sample(&self, from: Vec3, u: Vec2) -> Option<LightSample> {
        let p = self.p0 + u.x * self.edge_u + u.y * self.edge_v;
        let to_light = p - from;
        let dist2 = to_light.length_squared();
        if dist2 <= 0.0 {
            return None;
        }
        let dist = dist2.sqrt();
        let wi = to_light / dist;
        let cos_light = -wi.dot(self.normal);
        if cos_light <= 0.0 {
            return None;
        }
        // Convert area-domain pdf (1/area) to solid-angle pdf at `from`.
        let pdf = dist2 / (cos_light * self.area);
        Some(LightSample {
            position: p,
            wi,
            distance: dist,
            li: self.emission,
            pdf,
        })
    }

    fn pdf(&self, from: Vec3, wi: Vec3) -> f32 {
        let Some((t, _)) = self.intersect_plane(from, wi, 1e-4, f32::INFINITY) else {
            return 0.0;
        };
        let cos_light = -wi.dot(self.normal);
        if cos_light <= 0.0 {
            return 0.0;
        }
        t * t / (cos_light * self.area)
    }

    fn le(&self, _position: Vec3, normal: Vec3, wo: Vec3) -> Vec3 {
        if normal.dot(wo) > 0.0 {
            self.emission
        } else {
            Vec3::ZERO
        }
    }

    fn intersect(&self, ray: &Ray) -> Option<LightHit> {
        let (t, p) = self.intersect_plane(ray.origin, ray.direction, ray.t_min, ray.t_max)?;
        Some(LightHit {
            t,
            position: p,
            normal: self.normal,
        })
    }
}

// ============================================================================
// Environment lighting
// ============================================================================
//
// Environment lights live at infinity and contribute to a ray that escapes
// the scene's geometry. They satisfy a different contract than `Light`:
// there's no surface to intersect, and the integrator queries them in two
// places — when a ray misses, and as an extra NEE arm at every shading
// point. We therefore split the trait rather than overload `Light`.

#[derive(Debug, Clone, Copy)]
pub struct EnvSample {
    /// Unit world-space direction sampled toward the environment.
    pub wi: Vec3,
    /// Radiance arriving from `wi`.
    pub li: Vec3,
    /// Solid-angle pdf at the shading point. Environment lights are at
    /// infinity, so this is independent of the shading position.
    pub pdf: f32,
}

pub trait EnvironmentLight: Send + Sync {
    /// Radiance arriving from the unit direction `wi`. Used on miss and
    /// for the BSDF→env arm of MIS.
    fn le(&self, wi: Vec3) -> Vec3;
    /// Importance-sample a direction.
    fn sample(&self, u: Vec2) -> EnvSample;
    /// Solid-angle pdf at any shading point of having sampled `wi`.
    fn pdf(&self, wi: Vec3) -> f32;
}

/// Equirectangular HDRI environment light. Built once from a linear-RGB
/// pixel buffer; importance sampling uses a 2D piecewise-constant
/// distribution weighted by per-pixel luminance × `sin(theta)` so the
/// solid-angle Jacobian of the equirectangular map cancels out.
///
/// Coordinate convention: `+Y` is up. The mapping uses
/// `theta = acos(wi.y)` (0 = zenith, π = nadir) and
/// `phi = atan2(wi.z, wi.x)` (-π = -X, +π = -X again, wrapping). The
/// `intensity` multiplier scales radiance for artistic control without
/// rebuilding the texture.
pub struct HdriEnvironmentLight {
    width: u32,
    height: u32,
    pixels: Vec<Vec3>,
    distribution: Distribution2D,
    intensity: f32,
}

impl HdriEnvironmentLight {
    pub fn new(width: u32, height: u32, pixels: Vec<Vec3>, intensity: f32) -> Self {
        assert_eq!(
            pixels.len(),
            (width as usize) * (height as usize),
            "pixel buffer length must equal width * height"
        );
        let w = width as usize;
        let h = height as usize;
        let mut weights = Vec::with_capacity(w * h);
        for y in 0..h {
            // theta at the row centre, [0, π].
            let theta = (y as f32 + 0.5) / h as f32 * PI;
            let sin_theta = theta.sin().max(0.0);
            for x in 0..w {
                let p = pixels[y * w + x];
                // Rec. 709 luminance.
                let lum = 0.212_67 * p.x + 0.715_16 * p.y + 0.072_17 * p.z;
                weights.push(lum.max(0.0) * sin_theta);
            }
        }
        let distribution = Distribution2D::new(&weights, w, h);
        Self {
            width,
            height,
            pixels,
            distribution,
            intensity,
        }
    }

    /// Constant-radiance environment (uniform sky).
    pub fn constant(color: Vec3, intensity: f32) -> Self {
        Self::new(1, 1, vec![color], intensity)
    }

    /// Two-tone vertical sky: `zenith` straight up, `horizon` at the
    /// equator, dimmed below the horizon. Useful for sanity-checking env
    /// integration without an HDRI asset.
    pub fn gradient(zenith: Vec3, horizon: Vec3, intensity: f32) -> Self {
        let h = 64usize;
        let w = 128usize;
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            // 0 at top, 1 at bottom.
            let v = (y as f32 + 0.5) / h as f32;
            // cos(theta): 1 at top, -1 at bottom.
            let cos_theta = 1.0 - 2.0 * v;
            let color = if cos_theta >= 0.0 {
                horizon.lerp(zenith, cos_theta)
            } else {
                // Below the horizon: darken without inverting hue.
                horizon * (1.0 + cos_theta * 0.5).max(0.0)
            };
            for _ in 0..w {
                pixels.push(color);
            }
        }
        Self::new(w as u32, h as u32, pixels, intensity)
    }

    fn dir_to_uv(wi: Vec3) -> Vec2 {
        let theta = wi.y.clamp(-1.0, 1.0).acos();
        let phi = wi.z.atan2(wi.x);
        let u = (phi + PI) * FRAC_1_PI * 0.5;
        let v = theta * FRAC_1_PI;
        Vec2::new(u, v)
    }

    fn uv_to_dir(uv: Vec2) -> Vec3 {
        let phi = uv.x * TAU - PI;
        let theta = uv.y * PI;
        let sin_theta = theta.sin();
        Vec3::new(sin_theta * phi.cos(), theta.cos(), sin_theta * phi.sin())
    }

    /// Bilinear texture lookup with wrap-u, clamp-v.
    fn sample_texture(&self, uv: Vec2) -> Vec3 {
        let w = self.width as i32;
        let h = self.height as i32;
        let u = (uv.x.fract() + 1.0).fract();
        let v = uv.y.clamp(0.0, 1.0);
        let fx = u * w as f32 - 0.5;
        let fy = (v * h as f32 - 0.5).clamp(0.0, h as f32 - 1.0);
        let x0i = fx.floor() as i32;
        let y0i = fy.floor() as i32;
        let dx = fx - x0i as f32;
        let dy = fy - y0i as f32;
        let x0 = x0i.rem_euclid(w);
        let x1 = (x0 + 1).rem_euclid(w);
        let y0 = y0i.max(0).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let idx = |x: i32, y: i32| (y as usize) * (self.width as usize) + (x as usize);
        let p00 = self.pixels[idx(x0, y0)];
        let p10 = self.pixels[idx(x1, y0)];
        let p01 = self.pixels[idx(x0, y1)];
        let p11 = self.pixels[idx(x1, y1)];
        p00 * ((1.0 - dx) * (1.0 - dy))
            + p10 * (dx * (1.0 - dy))
            + p01 * ((1.0 - dx) * dy)
            + p11 * (dx * dy)
    }
}

impl EnvironmentLight for HdriEnvironmentLight {
    fn le(&self, wi: Vec3) -> Vec3 {
        self.sample_texture(Self::dir_to_uv(wi)) * self.intensity
    }

    fn sample(&self, u: Vec2) -> EnvSample {
        let (uv, pdf_uv) = self.distribution.sample_continuous(u);
        let wi = Self::uv_to_dir(uv);
        let theta = uv.y * PI;
        let sin_theta = theta.sin();
        // Convert (u, v) pdf to solid-angle pdf:
        //   du dv = (1/(2π)) dphi · (1/π) dtheta
        //   dω    = sin(theta) dphi dtheta
        //   pdf_ω = pdf_uv / (2π² sin θ)
        let pdf = if sin_theta > 1e-6 && pdf_uv > 0.0 {
            pdf_uv / (2.0 * PI * PI * sin_theta)
        } else {
            0.0
        };
        let li = self.sample_texture(uv) * self.intensity;
        EnvSample { wi, li, pdf }
    }

    fn pdf(&self, wi: Vec3) -> f32 {
        let uv = Self::dir_to_uv(wi);
        let theta = uv.y * PI;
        let sin_theta = theta.sin();
        if sin_theta < 1e-6 {
            return 0.0;
        }
        let pdf_uv = self.distribution.pdf_at(uv);
        pdf_uv / (2.0 * PI * PI * sin_theta)
    }
}
