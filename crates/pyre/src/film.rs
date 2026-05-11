use crate::integrator::PixelSample;
use glam::Vec3;
use rayon::prelude::*;
use std::path::Path;

/// Linear-light framebuffer. The renderer writes a full `PixelSample` per
/// pixel (radiance + AOVs); per-pass storage is `width * height` samples
/// of ~40 bytes each, traded off for a much simpler render loop than
/// optional-AOV buffers would require. PNG and EXR sinks read what they
/// need from this single buffer.
///
/// `save_aovs` controls which AOV layers `save_exr` writes to disk —
/// the integrator always emits them, so this is a sink-side choice, not
/// a render-side allocation question.
pub struct Film {
    width: u32,
    height: u32,
    samples: Vec<PixelSample>,
    save_aovs: AovSet,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AovSet {
    pub albedo: bool,
    pub normal: bool,
    pub depth: bool,
}

impl AovSet {
    pub const ALL: Self = Self {
        albedo: true,
        normal: true,
        depth: true,
    };
}

impl Film {
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            samples: vec![PixelSample::MISS; n],
            save_aovs: AovSet::default(),
        }
    }

    pub fn with_save_aovs(mut self, aovs: AovSet) -> Self {
        self.save_aovs = aovs;
        self
    }

    /// Wrap a pre-computed linear-radiance buffer as a `Film` so it can be
    /// fed to `save_png`. The buffer is row-major with origin at top-
    /// left, matching the convention used by `render`. AOVs are not
    /// represented — only the beauty channel is meaningful in this
    /// snapshot view. Used by the progressive viewer.
    pub fn from_buffer(width: u32, height: u32, pixels: Vec<Vec3>) -> Self {
        assert_eq!(
            pixels.len(),
            (width as usize) * (height as usize),
            "pixel buffer length must equal width * height"
        );
        let samples = pixels
            .into_iter()
            .map(|p| PixelSample {
                radiance: p,
                albedo: Vec3::ZERO,
                normal: Vec3::ZERO,
                depth: 0.0,
            })
            .collect();
        Self {
            width,
            height,
            samples,
            save_aovs: AovSet::default(),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn samples(&self) -> &[PixelSample] {
        &self.samples
    }

    /// Fill every pixel in parallel using `f(x, y)`. Pixel `(0, 0)` is the
    /// top-left of the image. The closure returns a full `PixelSample`;
    /// the film stores it directly.
    pub fn render<F>(&mut self, f: F)
    where
        F: Fn(u32, u32) -> PixelSample + Sync + Send,
    {
        let width = self.width;
        self.samples
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, sample) in row.iter_mut().enumerate() {
                    *sample = f(x as u32, y as u32);
                }
            });
    }

    /// Save each enabled AOV as a sidecar PNG visualisation alongside
    /// `base_path`. Beauty already lives at `base_path`; this method
    /// writes `<stem>.albedo.png` (gamma 2.2, clamped), `<stem>.normal.png`
    /// (linear `0.5 * (n + 1)` mapping), and `<stem>.depth.png` (min-max
    /// normalised, near = bright). No tone-mapping — just enough so the
    /// AOVs are inspectable in a normal image viewer.
    pub fn save_aov_pngs<P: AsRef<Path>>(&self, base_path: P) -> image::ImageResult<()> {
        let base = base_path.as_ref();
        let parent = base.parent().unwrap_or_else(|| Path::new("."));
        let stem = base
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("render");

        if self.save_aovs.albedo {
            let mut buf = image::RgbImage::new(self.width, self.height);
            for (i, s) in self.samples.iter().enumerate() {
                let x = (i as u32) % self.width;
                let y = (i as u32) / self.width;
                let e = s.albedo.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2) * 255.0;
                buf.put_pixel(x, y, image::Rgb([e.x as u8, e.y as u8, e.z as u8]));
            }
            buf.save(parent.join(format!("{stem}.albedo.png")))?;
        }

        if self.save_aovs.normal {
            let mut buf = image::RgbImage::new(self.width, self.height);
            for (i, s) in self.samples.iter().enumerate() {
                let x = (i as u32) % self.width;
                let y = (i as u32) / self.width;
                let e = (s.normal * 0.5 + Vec3::splat(0.5))
                    .clamp(Vec3::ZERO, Vec3::ONE)
                    * 255.0;
                buf.put_pixel(x, y, image::Rgb([e.x as u8, e.y as u8, e.z as u8]));
            }
            buf.save(parent.join(format!("{stem}.normal.png")))?;
        }

        if self.save_aovs.depth {
            let valid: Vec<f32> = self.samples.iter().map(|s| s.depth).collect();
            // Robust min/max ignoring miss-sentinels (depth = 0.0).
            let (mut lo, mut hi) = (f32::INFINITY, 0.0_f32);
            for &d in &valid {
                if d > 0.0 {
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
            }
            let range = (hi - lo).max(1e-6);
            let mut buf = image::GrayImage::new(self.width, self.height);
            for (i, s) in self.samples.iter().enumerate() {
                let x = (i as u32) % self.width;
                let y = (i as u32) / self.width;
                let v = if s.depth > 0.0 {
                    // Closer = brighter.
                    (1.0 - (s.depth - lo) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                buf.put_pixel(x, y, image::Luma([(v * 255.0) as u8]));
            }
            buf.save(parent.join(format!("{stem}.depth.png")))?;
        }

        Ok(())
    }

    /// Save the beauty channel to a PNG with gamma 2.2 encoding. Values
    /// outside `[0, 1]` are clamped — proper tonemapping comes with the
    /// `color` module.
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> image::ImageResult<()> {
        let mut buf = image::RgbImage::new(self.width, self.height);
        for (i, sample) in self.samples.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            let encoded = sample
                .radiance
                .clamp(Vec3::ZERO, Vec3::ONE)
                .powf(1.0 / 2.2)
                * 255.0;
            buf.put_pixel(
                x,
                y,
                image::Rgb([encoded.x as u8, encoded.y as u8, encoded.z as u8]),
            );
        }
        buf.save(path)
    }

    /// Save the beauty pass plus any enabled AOVs as a single-layer
    /// multi-channel EXR. Channel naming uses dotted prefixes
    /// (`albedo.R`, `albedo.G`, `albedo.B`, `N.X`, `N.Y`, `N.Z`, `Z`) so
    /// Nuke / Fusion auto-group them. Beauty stays as plain `R`/`G`/`B`
    /// for OIDN's default input layout. EXR is linear — no gamma encoding.
    pub fn save_exr<P: AsRef<Path>>(&self, path: P) -> Result<(), exr::error::Error> {
        use exr::prelude::*;
        use smallvec::smallvec;

        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;

        // Helper: split a Vec3 stream into three f32 planes.
        let split_rgb = |extract: &dyn Fn(&PixelSample) -> Vec3| {
            let mut r = vec![0f32; n];
            let mut g = vec![0f32; n];
            let mut b = vec![0f32; n];
            for (i, s) in self.samples.iter().enumerate() {
                let v = extract(s);
                r[i] = v.x;
                g[i] = v.y;
                b[i] = v.z;
            }
            (r, g, b)
        };

        let mut channels: smallvec::SmallVec<[AnyChannel<FlatSamples>; 4]> = smallvec![];

        // Beauty — plain R/G/B (OIDN default).
        let (br, bg, bb) = split_rgb(&|s| s.radiance);
        channels.push(AnyChannel::new("R", FlatSamples::F32(br)));
        channels.push(AnyChannel::new("G", FlatSamples::F32(bg)));
        channels.push(AnyChannel::new("B", FlatSamples::F32(bb)));

        if self.save_aovs.albedo {
            let (r, g, b) = split_rgb(&|s| s.albedo);
            channels.push(AnyChannel::new("albedo.R", FlatSamples::F32(r)));
            channels.push(AnyChannel::new("albedo.G", FlatSamples::F32(g)));
            channels.push(AnyChannel::new("albedo.B", FlatSamples::F32(b)));
        }
        if self.save_aovs.normal {
            let (x, y, z) = split_rgb(&|s| s.normal);
            channels.push(AnyChannel::new("N.X", FlatSamples::F32(x)));
            channels.push(AnyChannel::new("N.Y", FlatSamples::F32(y)));
            channels.push(AnyChannel::new("N.Z", FlatSamples::F32(z)));
        }
        if self.save_aovs.depth {
            let depth: Vec<f32> = self.samples.iter().map(|s| s.depth).collect();
            channels.push(AnyChannel::new("Z", FlatSamples::F32(depth)));
        }

        // EXR requires channel names sorted alphabetically. exr's `Text`
        // type doesn't expose `as_str` directly, but Display works.
        channels.sort_by(|a, b| a.name.to_string().cmp(&b.name.to_string()));
        let any_channels = AnyChannels { list: channels };
        let layer = Layer::new(
            (w, h),
            LayerAttributes::default(),
            Encoding::FAST_LOSSLESS,
            any_channels,
        );
        let image = Image::from_layer(layer);
        image.write().to_file(path)
    }
}
