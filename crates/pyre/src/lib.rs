//! Pyre — a high-fidelity offline path tracer.
//!
//! See `CLAUDE.md` at the workspace root for the architecture overview and
//! milestone roadmap. The module layout mirrors the responsibilities of a
//! textbook path tracer (PBRT chapter ordering); each module exposes a trait
//! plus implementations.

pub mod camera;
pub mod film;
pub mod geometry;
pub mod io;
pub mod math;

pub use camera::{Camera, PinholeCamera};
pub use film::Film;
pub use geometry::{Bvh, MeshInstance, Shape, Sphere, SurfaceInteraction, TriangleMesh};
pub use io::{GltfError, load_gltf};
pub use math::{Bounds3, Ray};
