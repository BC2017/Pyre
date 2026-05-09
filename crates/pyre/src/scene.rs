//! The renderable world. A `Scene` wraps geometry primitives (each tagged
//! with a material id), the materials they reference, and emissive lights.
//!
//! Lights live separately from primitives so shadow rays only test against
//! occluding geometry. When a path's BSDF-sampled ray happens to hit a light,
//! the integrator picks it up via `SceneHit::Light` and applies MIS.
//!
//! Milestone 3 keeps a flat `Vec<Primitive>` and iterates linearly. A
//! top-level BVH (TLAS) over instances is a milestone 5+ optimization.

use crate::{
    geometry::{MeshInstance, Shape, SurfaceInteraction},
    light::Light,
    material::Bsdf,
    math::{Bounds3, Ray},
};
use glam::Vec3;

pub struct Primitive {
    pub instance: MeshInstance,
    pub material_id: u32,
}

pub struct Scene {
    pub primitives: Vec<Primitive>,
    pub materials: Vec<Box<dyn Bsdf>>,
    pub lights: Vec<Box<dyn Light>>,
}

#[derive(Debug, Clone, Copy)]
pub enum HitKind {
    Surface { material_id: u32, primitive_id: u32 },
    Light { light_id: u32 },
}

pub struct SceneHit {
    pub interaction: SurfaceInteraction,
    pub kind: HitKind,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<SceneHit> {
        let mut closest: Option<SceneHit> = None;
        let mut t_max = ray.t_max;

        for (id, prim) in self.primitives.iter().enumerate() {
            let mut r = *ray;
            r.t_max = t_max;
            if let Some(it) = prim.instance.intersect(&r) {
                if it.t < t_max {
                    t_max = it.t;
                    closest = Some(SceneHit {
                        interaction: it,
                        kind: HitKind::Surface {
                            material_id: prim.material_id,
                            primitive_id: id as u32,
                        },
                    });
                }
            }
        }

        for (id, light) in self.lights.iter().enumerate() {
            let mut r = *ray;
            r.t_max = t_max;
            if let Some(lh) = light.intersect(&r) {
                if lh.t < t_max {
                    t_max = lh.t;
                    closest = Some(SceneHit {
                        interaction: SurfaceInteraction {
                            t: lh.t,
                            position: lh.position,
                            normal: lh.normal,
                        },
                        kind: HitKind::Light { light_id: id as u32 },
                    });
                }
            }
        }

        closest
    }

    /// Yes/no shadow ray test: is anything between `from` and `to`?
    /// Tests primitives only — lights aren't occluders.
    pub fn occluded(&self, from: Vec3, to: Vec3) -> bool {
        let to_from = to - from;
        let dist = to_from.length();
        if dist <= 0.0 {
            return false;
        }
        let dir = to_from / dist;
        let ray = Ray {
            origin: from,
            direction: dir,
            t_min: 1e-3,
            t_max: dist - 1e-3,
        };
        for prim in &self.primitives {
            if prim.instance.intersect(&ray).is_some() {
                return true;
            }
        }
        false
    }

    /// World-space AABB of all primitives. Lights are omitted because the
    /// auto-frame camera should size to visible geometry, not (for example)
    /// distant environment lights.
    pub fn bounds(&self) -> Bounds3 {
        let mut b = Bounds3::EMPTY;
        for prim in &self.primitives {
            b = b.union(&prim.instance.bvh.root_bounds());
        }
        b
    }

    pub fn triangle_count(&self) -> usize {
        self.primitives
            .iter()
            .map(|p| p.instance.mesh.triangle_count())
            .sum()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
