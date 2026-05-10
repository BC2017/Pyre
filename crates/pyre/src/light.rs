//! Light sources. The path tracer uses each light for two things:
//! - **direct sampling** (next-event estimation): pick a point on the light
//!   from a surface hit and weight by `Li / pdf` (with MIS against BSDF
//!   sampling);
//! - **incidental hits**: when a BSDF-sampled ray happens to hit the light
//!   geometry, the integrator pulls `Le` directly so the path tracer remains
//!   unbiased even without NEE.

use crate::math::Ray;
use glam::{Vec2, Vec3};

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
