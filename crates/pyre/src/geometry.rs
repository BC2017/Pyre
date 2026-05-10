use crate::math::Ray;
use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct SurfaceInteraction {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
}

pub trait Shape: Send + Sync {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction>;
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Shape for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction> {
        // Ray direction is normalized (Ray::new normalizes), so the quadratic
        // a-coefficient is 1.
        let oc = ray.origin - self.center;
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrt_d = discriminant.sqrt();

        let near = -half_b - sqrt_d;
        let far = -half_b + sqrt_d;
        let t = if near >= ray.t_min && near <= ray.t_max {
            near
        } else if far >= ray.t_min && far <= ray.t_max {
            far
        } else {
            return None;
        };

        let position = ray.at(t);
        let normal = (position - self.center) / self.radius;
        Some(SurfaceInteraction {
            t,
            position,
            normal,
        })
    }
}
