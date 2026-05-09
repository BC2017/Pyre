use crate::math::Ray;
use glam::Vec3;

pub trait Camera: Send + Sync {
    /// Generate a primary ray for a sample at NDC coordinates `(ndc_x, ndc_y)`
    /// in `[-1, 1]^2`. `(-1, -1)` is the bottom-left of the image and `(1, 1)`
    /// is the top-right.
    fn generate_ray(&self, ndc_x: f32, ndc_y: f32) -> Ray;
}

#[derive(Debug, Clone, Copy)]
pub struct PinholeCamera {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    half_width: f32,
    half_height: f32,
}

impl PinholeCamera {
    /// Build a right-handed look-at camera. `vfov_deg` is the full vertical
    /// field of view in degrees; `aspect` is image width / height.
    pub fn look_at(
        origin: Vec3,
        target: Vec3,
        world_up: Vec3,
        vfov_deg: f32,
        aspect: f32,
    ) -> Self {
        let forward = (target - origin).normalize();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward);
        let half_height = (vfov_deg.to_radians() / 2.0).tan();
        let half_width = aspect * half_height;
        Self {
            origin,
            forward,
            right,
            up,
            half_width,
            half_height,
        }
    }
}

impl Camera for PinholeCamera {
    fn generate_ray(&self, ndc_x: f32, ndc_y: f32) -> Ray {
        let direction = self.forward
            + ndc_x * self.half_width * self.right
            + ndc_y * self.half_height * self.up;
        Ray::new(self.origin, direction)
    }
}
