use glam::Vec3;
use rayon::prelude::*;
use std::path::Path;

/// Linear-light framebuffer. The renderer writes radiance values directly;
/// gamma encoding happens only at the PNG/JPG sink. EXR output (when added)
/// will write linear values verbatim.
pub struct Film {
    width: u32,
    height: u32,
    pixels: Vec<Vec3>,
}

impl Film {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![Vec3::ZERO; (width as usize) * (height as usize)],
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }

    /// Fill every pixel in parallel using `f(x, y)`. Pixel `(0, 0)` is the
    /// top-left of the image.
    pub fn render<F>(&mut self, f: F)
    where
        F: Fn(u32, u32) -> Vec3 + Sync + Send,
    {
        let width = self.width;
        self.pixels
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, pixel) in row.iter_mut().enumerate() {
                    *pixel = f(x as u32, y as u32);
                }
            });
    }

    /// Save the film to a PNG, applying a gamma 2.2 encoding. Values outside
    /// `[0, 1]` are clamped — proper tonemapping comes with the `color` module.
    pub fn save_png<P: AsRef<Path>>(&self, path: P) -> image::ImageResult<()> {
        let mut buf = image::RgbImage::new(self.width, self.height);
        for (i, pixel) in self.pixels.iter().enumerate() {
            let x = (i as u32) % self.width;
            let y = (i as u32) / self.width;
            let encoded = pixel.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2) * 255.0;
            buf.put_pixel(
                x,
                y,
                image::Rgb([encoded.x as u8, encoded.y as u8, encoded.z as u8]),
            );
        }
        buf.save(path)
    }
}
