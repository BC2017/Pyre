use glam::Vec3;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
    pub t_min: f32,
    pub t_max: f32,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            t_min: 1e-4,
            t_max: f32::INFINITY,
        }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Bounds3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds3 {
    pub const EMPTY: Self = Self {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn point(p: Vec3) -> Self {
        Self { min: p, max: p }
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn extend(&self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        2.0 * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    pub fn centroid(&self) -> Vec3 {
        0.5 * (self.min + self.max)
    }
}

/// Orthonormal basis with `n` as the +Z axis. Used to transform between world
/// coordinates and BSDF-local coordinates where the shading normal points up.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub t: Vec3,
    pub bt: Vec3,
    pub n: Vec3,
}

impl Frame {
    pub fn from_normal(n: Vec3) -> Self {
        // Pick a helper axis that isn't (nearly) parallel to n, then
        // Gram–Schmidt to build a tangent and bitangent.
        let helper = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
        let t = helper.cross(n).normalize();
        let bt = n.cross(t);
        Frame { t, bt, n }
    }

    pub fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.t), v.dot(self.bt), v.dot(self.n))
    }

    pub fn to_world(&self, v: Vec3) -> Vec3 {
        v.x * self.t + v.y * self.bt + v.z * self.n
    }
}
