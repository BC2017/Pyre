use crate::distribution::concentric_disk;
use crate::math::Ray;
use glam::{Vec2, Vec3};

pub trait Camera: Send + Sync {
    /// Generate a primary ray for a sample at NDC coordinates `(ndc_x, ndc_y)`
    /// in `[-1, 1]^2`. `(-1, -1)` is the bottom-left of the image and `(1, 1)`
    /// is the top-right. `lens_sample` is a uniform `[0,1)^2` sample used by
    /// aperture-bearing cameras (`ThinLensCamera`); pinhole cameras ignore it.
    fn generate_ray(&self, ndc_x: f32, ndc_y: f32, lens_sample: Vec2) -> Ray;
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
    fn generate_ray(&self, ndc_x: f32, ndc_y: f32, _lens_sample: Vec2) -> Ray {
        let direction = self.forward
            + ndc_x * self.half_width * self.right
            + ndc_y * self.half_height * self.up;
        Ray::new(self.origin, direction)
    }
}

/// Thin-lens camera (Potmesil & Chakravarty 1981). Models a finite-aperture
/// lens that focuses sharply on a single plane and progressively defocuses
/// objects in front of and behind it. Aperture is sampled by concentric
/// disk mapping so stratified samplers retain low discrepancy.
///
/// The pinhole image stays the same — `ThinLensCamera` re-uses the
/// PinholeCamera ray direction to find a focal point on the focus plane,
/// then perturbs the origin by a lens offset and re-aims through the
/// focal point. As `aperture_radius → 0` this degenerates to a pinhole.
#[derive(Debug, Clone, Copy)]
pub struct ThinLensCamera {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    half_width: f32,
    half_height: f32,
    aperture_radius: f32,
    focus_distance: f32,
}

impl ThinLensCamera {
    pub fn look_at(
        origin: Vec3,
        target: Vec3,
        world_up: Vec3,
        vfov_deg: f32,
        aspect: f32,
        aperture_radius: f32,
        focus_distance: f32,
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
            aperture_radius,
            focus_distance,
        }
    }
}

impl Camera for ThinLensCamera {
    fn generate_ray(&self, ndc_x: f32, ndc_y: f32, lens_sample: Vec2) -> Ray {
        let pinhole_dir = (self.forward
            + ndc_x * self.half_width * self.right
            + ndc_y * self.half_height * self.up)
            .normalize();
        // Distance along the pinhole ray to the focus plane (which is
        // parallel to the image plane, offset by `focus_distance` along
        // `forward`).
        let focal_t = self.focus_distance / pinhole_dir.dot(self.forward);
        let focal_point = self.origin + focal_t * pinhole_dir;

        let disk = concentric_disk(lens_sample);
        let lens_offset = self.aperture_radius * (disk.x * self.right + disk.y * self.up);
        let lens_origin = self.origin + lens_offset;
        let dir = (focal_point - lens_origin).normalize();
        Ray::new(lens_origin, dir)
    }
}
