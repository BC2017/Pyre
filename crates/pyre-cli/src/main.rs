use anyhow::{Context, Result};
use clap::Parser;
use glam::Vec3;
use pyre::{
    Bounds3, Camera, DiffuseAreaQuadLight, DisneyBsdf, Film, IndependentSampler, Lambertian,
    MeshInstance, PathIntegrator, PinholeCamera, Primitive, Sampler, Scene, TriangleMesh,
    load_gltf, pixel_seed,
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

    /// Path to a glTF/.glb scene. If omitted, renders the procedural Cornell box.
    #[arg(short, long)]
    scene: Option<PathBuf>,

    #[arg(long, default_value_t = 800)]
    width: u32,

    #[arg(long, default_value_t = 600)]
    height: u32,

    /// Samples per pixel.
    #[arg(long, default_value_t = 16)]
    spp: u32,

    /// Maximum path depth (forced termination — Russian roulette typically
    /// kills paths earlier).
    #[arg(long, default_value_t = 8)]
    max_depth: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let args = Cli::parse();
    let aspect = args.width as f32 / args.height as f32;

    let scene = if let Some(path) = &args.scene {
        let meshes = load_gltf(path)
            .with_context(|| format!("loading glTF from {}", path.display()))?;
        tracing::info!(primitives = meshes.len(), "glTF loaded");
        let mut scene = Scene::new();
        scene
            .materials
            .push(Box::new(Lambertian { albedo: Vec3::splat(0.73) }));
        for mesh in meshes {
            scene.primitives.push(Primitive {
                instance: MeshInstance::build(mesh),
                material_id: 0,
            });
        }
        scene
    } else {
        cornell_box()
    };

    tracing::info!(
        triangles = scene.triangle_count(),
        materials = scene.materials.len(),
        lights = scene.lights.len(),
        "scene built"
    );

    let camera = if args.scene.is_some() {
        let b = scene.bounds();
        let diag = (b.max - b.min).length();
        if diag.is_finite() && diag > 0.0 {
            auto_frame_camera(b, 45.0, aspect)
        } else {
            PinholeCamera::look_at(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y, 45.0, aspect)
        }
    } else {
        // Cornell box: camera just outside the open front face, looking inward.
        PinholeCamera::look_at(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y, 45.0, aspect)
    };

    let integrator = PathIntegrator {
        max_depth: args.max_depth,
        min_rr_depth: 3,
    };

    let mut film = Film::new(args.width, args.height);
    let width = args.width;
    let height = args.height;
    let spp = args.spp;

    let started = Instant::now();
    film.render(|x, y| {
        let mut accum = Vec3::ZERO;
        for s in 0..spp {
            let mut sampler = IndependentSampler::new(pixel_seed(x, y, s));
            // Jitter within the pixel for free anti-aliasing.
            let jitter_x = sampler.next_f32();
            let jitter_y = sampler.next_f32();
            let ndc_x = 2.0 * (x as f32 + jitter_x) / width as f32 - 1.0;
            let ndc_y = 1.0 - 2.0 * (y as f32 + jitter_y) / height as f32;
            let ray = camera.generate_ray(ndc_x, ndc_y);
            accum += integrator.li(ray, &scene, &mut sampler);
        }
        accum / spp as f32
    });
    let elapsed = started.elapsed();

    film.save_png(&args.output)?;
    tracing::info!(
        output = %args.output.display(),
        elapsed_ms = elapsed.as_millis() as u64,
        triangles = scene.triangle_count(),
        spp = args.spp,
        pixels = args.width * args.height,
        "render complete"
    );
    Ok(())
}

fn auto_frame_camera(bounds: Bounds3, vfov_deg: f32, aspect: f32) -> PinholeCamera {
    let center = bounds.centroid();
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

fn quad_mesh(corners: [Vec3; 4], normal: Vec3) -> TriangleMesh {
    TriangleMesh {
        positions: corners.to_vec(),
        indices: vec![0, 1, 2, 0, 2, 3],
        normals: Some(vec![normal; 4]),
        uvs: None,
    }
}

/// Classic Cornell box: 5 walls (no front), one ceiling area light, plus a
/// white sphere on the floor for visual interest. Centered at the origin
/// with half-extent 1.
fn cornell_box() -> Scene {
    let mut scene = Scene::new();

    let white = scene.materials.len() as u32;
    scene
        .materials
        .push(Box::new(Lambertian { albedo: Vec3::splat(0.73) }));
    let red = scene.materials.len() as u32;
    scene.materials.push(Box::new(Lambertian {
        albedo: Vec3::new(0.65, 0.05, 0.05),
    }));
    let green = scene.materials.len() as u32;
    scene.materials.push(Box::new(Lambertian {
        albedo: Vec3::new(0.12, 0.45, 0.15),
    }));

    let s = 1.0;

    // Floor — normal +Y
    let floor = quad_mesh(
        [
            Vec3::new(-s, -s, -s),
            Vec3::new(s, -s, -s),
            Vec3::new(s, -s, s),
            Vec3::new(-s, -s, s),
        ],
        Vec3::Y,
    );
    // Ceiling — normal -Y
    let ceiling = quad_mesh(
        [
            Vec3::new(-s, s, -s),
            Vec3::new(s, s, -s),
            Vec3::new(s, s, s),
            Vec3::new(-s, s, s),
        ],
        -Vec3::Y,
    );
    // Back wall — normal +Z (toward camera)
    let back = quad_mesh(
        [
            Vec3::new(-s, -s, -s),
            Vec3::new(s, -s, -s),
            Vec3::new(s, s, -s),
            Vec3::new(-s, s, -s),
        ],
        Vec3::Z,
    );
    // Left wall (red) — normal +X
    let left = quad_mesh(
        [
            Vec3::new(-s, -s, -s),
            Vec3::new(-s, -s, s),
            Vec3::new(-s, s, s),
            Vec3::new(-s, s, -s),
        ],
        Vec3::X,
    );
    // Right wall (green) — normal -X
    let right = quad_mesh(
        [
            Vec3::new(s, -s, -s),
            Vec3::new(s, s, -s),
            Vec3::new(s, s, s),
            Vec3::new(s, -s, s),
        ],
        -Vec3::X,
    );

    for (mesh, material_id) in [
        (floor, white),
        (ceiling, white),
        (back, white),
        (left, red),
        (right, green),
    ] {
        scene.primitives.push(Primitive {
            instance: MeshInstance::build(mesh),
            material_id,
        });
    }

    // Polished gold metal sphere (front-right).
    let gold = scene.materials.len() as u32;
    scene.materials.push(Box::new(DisneyBsdf {
        base_color: Vec3::new(1.0, 0.766, 0.336),
        metallic: 1.0,
        roughness: 0.1,
        specular: 0.5,
    }));
    let gold_sphere = make_uv_sphere(Vec3::new(0.35, -0.65, 0.25), 0.35, 48, 24);
    scene.primitives.push(Primitive {
        instance: MeshInstance::build(gold_sphere),
        material_id: gold,
    });

    // Glossy white plastic sphere (back-left).
    let plastic = scene.materials.len() as u32;
    scene.materials.push(Box::new(DisneyBsdf {
        base_color: Vec3::splat(0.72),
        metallic: 0.0,
        roughness: 0.18,
        specular: 0.5,
    }));
    let plastic_sphere = make_uv_sphere(Vec3::new(-0.35, -0.65, -0.3), 0.35, 48, 24);
    scene.primitives.push(Primitive {
        instance: MeshInstance::build(plastic_sphere),
        material_id: plastic,
    });

    // Ceiling area light. Cross product (edge_u × edge_v) points -Y so the
    // emissive face shines downward into the room.
    scene.lights.push(Box::new(DiffuseAreaQuadLight::new(
        Vec3::new(-0.3, 0.999, -0.3),
        Vec3::new(0.6, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.6),
        Vec3::splat(15.0),
    )));

    scene
}

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
