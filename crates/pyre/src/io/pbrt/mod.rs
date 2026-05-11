//! PBRT scene-file loader. Reads a `.pbrt` text file and produces a
//! `Scene` plus optional camera / film settings.
//!
//! Implementation is split across three submodules — `lexer` (tokens),
//! `parser` (token stream → flat directive AST), and `builder`
//! (directive list → `Scene`). The split keeps each layer small and
//! independently testable.
//!
//! ## Supported directive subset
//!
//! Render settings: `LookAt`, `Camera "perspective"`, `Film "image"`,
//! `Sampler`, `Integrator`, `Accelerator`, `PixelFilter` (last four
//! parsed but ignored — Pyre brings its own).
//!
//! Transforms: `Identity`, `Translate`, `Scale`, `Rotate`, `Transform`,
//! `ConcatTransform`. Stack manipulation: `AttributeBegin/End`,
//! `TransformBegin/End`, `WorldBegin/End`.
//!
//! Materials: `Material`, `MakeNamedMaterial`, `NamedMaterial`. Types:
//! `matte` → `Lambertian`, `plastic`/`metal`/`conductor` → `DisneyBsdf`.
//!
//! Lights: `LightSource "infinite"` (constant `L` or `mapname` HDRI),
//! `AreaLightSource "diffuse"` attached to a four-vertex two-triangle
//! quad → `DiffuseAreaQuadLight`.
//!
//! Shapes: `Shape "sphere"`, `Shape "trianglemesh"`.
//!
//! Anything else is logged via `tracing` and skipped so partial loads
//! still produce a renderable scene during bring-up.
//!
//! ## Coordinate system
//!
//! Pyre uses right-handed +Y up; PBRT's spec is +Z up by default. PBRT
//! files written for +Z up will load upside-down. The hand-authored
//! `scenes/cornell.pbrt` is written for +Y up so it round-trips the
//! built-in Cornell box exactly.

pub mod builder;
pub mod lexer;
pub mod parser;

pub use builder::{BuildError, CameraSpec, FilmSpec, LoadedPbrt};
pub use lexer::LexError;
pub use parser::ParseError;

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum PbrtError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("lex: {0}")]
    Lex(#[from] LexError),
    #[error("parse: {0}")]
    Parse(#[from] ParseError),
    #[error("build: {0}")]
    Build(#[from] BuildError),
}

pub fn load_pbrt<P: AsRef<Path>>(path: P) -> Result<LoadedPbrt, PbrtError> {
    let path = path.as_ref();
    let src = std::fs::read_to_string(path)?;
    let base_dir = path.parent().unwrap_or(Path::new("."));
    let toks = lexer::tokenize(&src)?;
    let dirs = parser::parse(&toks)?;
    let loaded = builder::build(dirs, base_dir)?;
    tracing::info!(
        path = %path.display(),
        primitives = loaded.scene.primitives.len(),
        materials = loaded.scene.materials.len(),
        lights = loaded.scene.lights.len(),
        env = loaded.scene.env.is_some(),
        "PBRT scene loaded"
    );
    Ok(loaded)
}
