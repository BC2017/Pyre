//! Pyre — a high-fidelity offline path tracer.
//!
//! See `CLAUDE.md` at the workspace root for the architecture overview and
//! milestone roadmap. The module layout mirrors the responsibilities of a
//! textbook path tracer (PBRT chapter ordering); each module exposes a trait
//! plus implementations.

pub mod camera;
pub mod film;
pub mod geometry;
pub mod integrator;
pub mod io;
pub mod light;
pub mod material;
pub mod math;
pub mod sampler;
pub mod scene;

pub use camera::{Camera, PinholeCamera};
pub use film::Film;
pub use geometry::{Bvh, MeshInstance, Shape, Sphere, SurfaceInteraction, TriangleMesh};
pub use integrator::PathIntegrator;
pub use io::{GltfError, load_gltf};
pub use light::{DiffuseAreaQuadLight, Light, LightHit, LightSample};
pub use material::{Bsdf, BsdfSample, DisneyBsdf, Lambertian};
pub use math::{Bounds3, Frame, Ray};
pub use sampler::{IndependentSampler, Sampler, pixel_seed};
pub use scene::{HitKind, Primitive, Scene, SceneHit};
