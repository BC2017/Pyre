//! Radiance .hdr loader. EXR support can land alongside the EXR writer
//! at milestone 8 — for now the workspace policy is: if you have an EXR
//! HDRI, convert it to .hdr (one `oiiotool` invocation) before loading.

use crate::light::HdriEnvironmentLight;
use glam::Vec3;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HdrLoadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(#[from] image::ImageError),
    #[error("unsupported environment-map extension: {0}")]
    UnsupportedExtension(String),
}

/// Load an HDRI environment map from disk. Currently supports `.hdr`
/// (Radiance RGBE). The returned environment has the supplied
/// `intensity` multiplier baked in for the `le`/`sample` results.
pub fn load_hdri<P: AsRef<Path>>(
    path: P,
    intensity: f32,
) -> Result<HdriEnvironmentLight, HdrLoadError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "hdr" | "pic" => load_radiance(path, intensity),
        other => Err(HdrLoadError::UnsupportedExtension(other.to_string())),
    }
}

fn load_radiance(path: &Path, intensity: f32) -> Result<HdriEnvironmentLight, HdrLoadError> {
    let img = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?
        .into_rgb32f();
    let (w, h) = img.dimensions();
    let pixels: Vec<Vec3> = img
        .pixels()
        .map(|p| Vec3::new(p[0], p[1], p[2]))
        .collect();
    Ok(HdriEnvironmentLight::new(w, h, pixels, intensity))
}
