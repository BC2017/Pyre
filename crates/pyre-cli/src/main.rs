use anyhow::{Context, Result};
use clap::Parser;
use glam::Vec3;
use pyre::{
    Bounds3, Camera, Film, MeshInstance, PinholeCamera, Ray, Shape, SurfaceInteraction,
    TriangleMesh, load_gltf,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing_subscriber::EnvFilter;

/// Pyre — a high-fidelity offline path tracer.
#[derive(Parser, Debug)]
#[command(name = "pyre", version, about)]
struct Cli {
    /// Output image path (PNG).
    #[arg(short, long, default_value = "out.png")]
    output: PathBuf,

    /// Path to a glTF/.glb scene. If omitted, a procedural test scene
    /// (UV sphere + ground plane) is rendered.
    #[arg(short, long)]
    scene: Option<PathBuf>,

    #[arg(long, default_value_t = 800)]
    width: u32,

    #[arg(long, default_value_t = 600)]
    height: u32,
}

/// Flat list of mesh instances. A proper Scene type with materials, lights,
/// and a top-level BVH lands in milestone 3+.
struct World {
    instances: Vec<MeshInstance>,
}

impl World {
    fn intersect(&self, ray: &Ray) -> Option<SurfaceInteraction> {
        let mut closest: Option<SurfaceInteraction> = None;
        let mut t_max = ray.t_max;
        for inst in &self.instances {
            let mut r = *ray;
            r.t_max = t_max;
            if let Some(hit) = inst.intersect(&r) {
                if hit.t < t_max {
                    t_max = hit.t;
                    closest = Some(hit);
                }
            }
        }
        closest
    }

    fn triangle_count(&self) -> usize {
        self.instances.iter().map(|i| i.mesh.triangle_count()).sum()
    }

    fn bounds(&self) -> Bounds3 {
        let mut b = Bounds3::EMPTY;
        for inst in &self.instances {
            b = b.union(&inst.bvh.root_bounds());
        }
        b
    }
}

/// Position the camera so the world bounds fit within the view frustum, with
/// a small margin. Camera looks at the bounds centroid from front-and-slightly
/// above (+Z, +Y).
fn auto_frame_camera(bounds: Bounds3, vfov_deg: f32, aspect: f32) -> PinholeCamera {
    let center = bounds.centroid();
    // Bounding-sphere radius from the AABB diagonal — slightly conservative
    // (over-estimates for non-cubic boxes), which is fine for safe framing.
    let radius = (bounds.max - bounds.min).length() * 0.5;
    let radius = radius.max(1e-3);

    let half_vfov = vfov_deg.to_radians() * 0.5;
    let half_hfov = (aspect * half_vfov.tan()).atan();
    let dist_v = radius / half_vfov.sin();
    let dist_h = radius / half_hfov.sin();
    let distance = dist_v.max(dist_h) * 1.15;

    let dir = Vec3::new(0.0, 0.3, 1.0).normalize();
    let origin = center + distance * dir;
    PinholeCamera::look_at(origin, center, Vec3::Y, vfov_deg, aspect)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let args = Cli::parse();
    let aspect = args.width as f32 / args.height as f32;

    let world = if let Some(path) = &args.scene {
        let meshes = load_gltf(path)
            .with_context(|| format!("loading glTF from {}", path.display()))?;
        tracing::info!(primitives = meshes.len(), "glTF loaded");
        World {
            instances: meshes.into_iter().map(MeshInstance::build).collect(),
        }
    } else {
        let sphere = make_uv_sphere(Vec3::ZERO, 1.0, 64, 32);
        let ground = ground_plane(-1.0, 5.0);
        World {
            instances: vec![MeshInstance::build(sphere), MeshInstance::build(ground)],
        }
    };

    tracing::info!(triangles = world.triangle_count(), "world built");

    let camera = if args.scene.is_some() {
        let b = world.bounds();
        let diag = (b.max - b.min).length();
        if diag.is_finite() && diag > 0.0 {
            auto_frame_camera(b, 45.0, aspect)
        } else {
            // Empty scene: fall back to a default camera.
            PinholeCamera::look_at(
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::ZERO,
                Vec3::Y,
                45.0,
                aspect,
            )
        }
    } else {
        // Procedural scene is hand-framed.
        PinholeCamera::look_at(
            Vec3::new(0.0, 0.7, 3.0),
            Vec3::ZERO,
            Vec3::Y,
            45.0,
            aspect,
        )
    };

    let mut film = Film::new(args.width, args.height);
    let started = Instant::now();
    film.render(|x, y| {
        let ndc_x = 2.0 * (x as f32 + 0.5) / args.width as f32 - 1.0;
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / args.height as f32;
        let ray = camera.generate_ray(ndc_x, ndc_y);
        if let Some(hit) = world.intersect(&ray) {
            // Milestone 2: shade by surface normal mapped into [0, 1].
            hit.normal * 0.5 + Vec3::splat(0.5)
        } else {
            let t = 0.5 * (ray.direction.y + 1.0);
            Vec3::ONE.lerp(Vec3::new(0.5, 0.7, 1.0), t)
        }
    });
    let elapsed = started.elapsed();

    film.save_png(&args.output)?;
    tracing::info!(
        output = %args.output.display(),
        elapsed_ms = elapsed.as_millis() as u64,
        triangles = world.triangle_count(),
        pixels = args.width * args.height,
        "render complete"
    );
    Ok(())
}

/// Latitude/longitude tessellation. Per-vertex normals are exact spherical
/// normals so the BVH+triangle code matches the analytic sphere visually.
fn make_uv_sphere(center: Vec3, radius: f32, u_segments: u32, v_segments: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for v in 0..=v_segments {
        let phi = std::f32::consts::PI * v as f32 / v_segments as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for u in 0..=u_segments {
            let theta = std::f32::consts::TAU * u as f32 / u_segments as f32;
            let n = Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin());
            positions.push(center + radius * n);
            normals.push(n);
        }
    }

    let row = u_segments + 1;
    for v in 0..v_segments {
        for u in 0..u_segments {
            let i00 = v * row + u;
            let i01 = v * row + u + 1;
            let i10 = (v + 1) * row + u;
            let i11 = (v + 1) * row + u + 1;
            indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }

    TriangleMesh {
        positions,
        indices,
        normals: Some(normals),
        uvs: None,
    }
}

fn ground_plane(y: f32, half_size: f32) -> TriangleMesh {
    let positions = vec![
        Vec3::new(-half_size, y, -half_size),
        Vec3::new(half_size, y, -half_size),
        Vec3::new(half_size, y, half_size),
        Vec3::new(-half_size, y, half_size),
    ];
    // CCW from above so the geometric normal also points +Y, matching the
    // explicit per-vertex normals.
    let indices = vec![0, 3, 2, 0, 2, 1];
    let normals = Some(vec![Vec3::Y; 4]);
    TriangleMesh {
        positions,
        indices,
        normals,
        uvs: None,
    }
}
