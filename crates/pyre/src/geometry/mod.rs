use crate::math::Ray;
use glam::Vec3;

pub mod bvh;
pub mod mesh;
pub mod sphere;

pub use bvh::{Bvh, BvhNode};
pub use mesh::{MeshInstance, TriangleMesh};
pub use sphere::Sphere;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInteraction {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
}

pub trait Shape: Send + Sync {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction>;
}
