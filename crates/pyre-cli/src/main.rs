use anyhow::Result;
use clap::Parser;
use glam::Vec3;
use pyre::{Camera, Film, PinholeCamera, Shape, Sphere};
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

    #[arg(long, default_value_t = 800)]
    width: u32,

    #[arg(long, default_value_t = 600)]
    height: u32,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let args = Cli::parse();
    let aspect = args.width as f32 / args.height as f32;

    let camera = PinholeCamera::look_at(
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        45.0,
        aspect,
    );
    let sphere = Sphere {
        center: Vec3::ZERO,
        radius: 1.0,
    };

    let mut film = Film::new(args.width, args.height);

    let started = Instant::now();
    film.render(|x, y| {
        let ndc_x = 2.0 * (x as f32 + 0.5) / args.width as f32 - 1.0;
        let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / args.height as f32;
        let ray = camera.generate_ray(ndc_x, ndc_y);
        if let Some(hit) = sphere.intersect(&ray) {
            // Milestone 1: shade by surface normal mapped into [0, 1].
            hit.normal * 0.5 + Vec3::splat(0.5)
        } else {
            // Sky gradient — placeholder until env lights land in milestone 6.
            let t = 0.5 * (ray.direction.y + 1.0);
            Vec3::ONE.lerp(Vec3::new(0.5, 0.7, 1.0), t)
        }
    });
    let elapsed = started.elapsed();

    film.save_png(&args.output)?;
    tracing::info!(
        output = %args.output.display(),
        elapsed_ms = elapsed.as_millis() as u64,
        pixels = args.width * args.height,
        "render complete"
    );
    Ok(())
}
