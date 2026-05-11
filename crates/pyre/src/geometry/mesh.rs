use super::{Bvh, Shape, SurfaceInteraction};
use crate::math::{Bounds3, Ray};
use glam::{Vec2, Vec3};

#[derive(Debug, Clone)]
pub struct TriangleMesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    pub normals: Option<Vec<Vec3>>,
    pub uvs: Option<Vec<Vec2>>,
}

impl TriangleMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn triangle_indices(&self, tri: u32) -> [u32; 3] {
        let i = tri as usize * 3;
        [self.indices[i], self.indices[i + 1], self.indices[i + 2]]
    }

    pub fn triangle_positions(&self, tri: u32) -> [Vec3; 3] {
        let [a, b, c] = self.triangle_indices(tri);
        [
            self.positions[a as usize],
            self.positions[b as usize],
            self.positions[c as usize],
        ]
    }

    pub fn triangle_bounds(&self, tri: u32) -> Bounds3 {
        let [v0, v1, v2] = self.triangle_positions(tri);
        Bounds3::point(v0).extend(v1).extend(v2)
    }

    pub fn triangle_centroid(&self, tri: u32) -> Vec3 {
        let [v0, v1, v2] = self.triangle_positions(tri);
        (v0 + v1 + v2) / 3.0
    }
}

/// Linear translation animation for an instance. `offset_t0` is the
/// world-space position at shutter open (time = 0), `offset_t1` at
/// shutter close (time = 1). Static instances skip this entirely.
#[derive(Debug, Clone, Copy)]
pub struct InstanceMotion {
    pub offset_t0: Vec3,
    pub offset_t1: Vec3,
}

impl InstanceMotion {
    pub fn at(&self, time: f32) -> Vec3 {
        self.offset_t0.lerp(self.offset_t1, time.clamp(0.0, 1.0))
    }

    pub fn swept_translation_bounds(&self) -> Bounds3 {
        Bounds3::point(self.offset_t0).extend(self.offset_t1)
    }
}

/// A mesh paired with its acceleration structure. The renderer always
/// intersects against this, never against a bare `TriangleMesh`.
///
/// `motion` adds translation-only motion blur: the underlying BVH stays
/// in static "object space" (the mesh's original positions), and the
/// `Shape` impl shifts incoming rays by `-motion.at(ray.time)` before the
/// BVH lookup and shifts the hit position back after. Rotation /
/// scaling motion is deferred until the first scene that needs it.
pub struct MeshInstance {
    pub mesh: TriangleMesh,
    pub bvh: Bvh,
    pub motion: Option<InstanceMotion>,
}

impl MeshInstance {
    pub fn build(mesh: TriangleMesh) -> Self {
        let n = mesh.triangle_count();
        let mut bounds = Vec::with_capacity(n);
        let mut centroids = Vec::with_capacity(n);
        for tri in 0..n as u32 {
            bounds.push(mesh.triangle_bounds(tri));
            centroids.push(mesh.triangle_centroid(tri));
        }
        let bvh = Bvh::build(&bounds, &centroids);
        MeshInstance { mesh, bvh, motion: None }
    }

    pub fn build_animated(mesh: TriangleMesh, motion: InstanceMotion) -> Self {
        let mut me = Self::build(mesh);
        me.motion = Some(motion);
        me
    }

    /// Swept world-space bounds over the full shutter `[0, 1]`. For static
    /// instances this is exactly the BVH's root bounds.
    pub fn world_bounds(&self) -> Bounds3 {
        let base = self.bvh.root_bounds();
        if let Some(m) = self.motion {
            let b0 = Bounds3::new(base.min + m.offset_t0, base.max + m.offset_t0);
            let b1 = Bounds3::new(base.min + m.offset_t1, base.max + m.offset_t1);
            b0.union(&b1)
        } else {
            base
        }
    }
}

impl Shape for MeshInstance {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction> {
        // Translate the ray into the static (offset = 0) frame of the
        // mesh. Translation-only motion preserves distance and direction,
        // so `t` and `normal` come straight back; we only need to shift
        // the hit position to world space.
        let offset = self.motion.map(|m| m.at(ray.time)).unwrap_or(Vec3::ZERO);
        let local_ray = if offset == Vec3::ZERO {
            *ray
        } else {
            Ray {
                origin: ray.origin - offset,
                ..*ray
            }
        };
        let mut it = self.bvh.intersect(&local_ray, |tri, r| {
            let [a, b, c] = self.mesh.triangle_indices(tri);
            let v0 = self.mesh.positions[a as usize];
            let v1 = self.mesh.positions[b as usize];
            let v2 = self.mesh.positions[c as usize];
            let (t, u, v) = moller_trumbore(r, v0, v1, v2)?;
            let position = r.at(t);
            let normal = if let Some(ns) = &self.mesh.normals {
                let w = 1.0 - u - v;
                (w * ns[a as usize] + u * ns[b as usize] + v * ns[c as usize]).normalize()
            } else {
                (v1 - v0).cross(v2 - v0).normalize()
            };
            Some(SurfaceInteraction {
                t,
                position,
                normal,
            })
        })?;
        if offset != Vec3::ZERO {
            it.position += offset;
        }
        Some(it)
    }
}

/// Möller–Trumbore ray-triangle intersection. Returns `(t, u, v)` where
/// `(u, v)` are the barycentric coordinates of vertices `v1` and `v2`
/// respectively (so vertex `v0`'s weight is `1 - u - v`).
fn moller_trumbore(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, f32, f32)> {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);
    if a.abs() < 1e-8 {
        return None;
    }
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * edge2.dot(q);
    if t > ray.t_min && t < ray.t_max {
        Some((t, u, v))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xy_triangle() -> TriangleMesh {
        TriangleMesh {
            positions: vec![
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            indices: vec![0, 1, 2],
            normals: None,
            uvs: None,
        }
    }

    #[test]
    fn hits_triangle_along_minus_z() {
        let inst = MeshInstance::build(xy_triangle());
        let ray = Ray::new(Vec3::new(0.0, 0.3, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = inst.intersect(&ray).expect("should hit");
        assert!((hit.t - 1.0).abs() < 1e-4);
        assert!(hit.normal.dot(Vec3::Z) > 0.99);
    }

    #[test]
    fn misses_triangle_below() {
        let inst = MeshInstance::build(xy_triangle());
        let ray = Ray::new(Vec3::new(0.0, -1.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(inst.intersect(&ray).is_none());
    }

    #[test]
    fn bvh_finds_closest_of_two() {
        // Two triangles parallel in xy, one at z=0 and another at z=-1.
        // A ray from +Z should hit the front (z=0) one first.
        let mesh = TriangleMesh {
            positions: vec![
                Vec3::new(-1.0, -1.0, 0.0),
                Vec3::new(1.0, -1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(0.0, 1.0, -1.0),
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
            normals: None,
            uvs: None,
        };
        let inst = MeshInstance::build(mesh);
        let ray = Ray::new(Vec3::new(0.0, 0.0, 2.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = inst.intersect(&ray).expect("should hit");
        assert!((hit.t - 2.0).abs() < 1e-4, "expected t≈2, got {}", hit.t);
    }
}
