# Pyre

Pyre is a CPU-first offline path tracer written in Rust. It is built as a
serious renderer architecture exercise: physically based light transport,
multiple importance sampling, Disney-style materials, acceleration structures,
progressive preview, and a trait-based design that can grow toward larger scene
formats and a future GPU backend.

The repository is currently named `rust-engine` for historical reasons; the
renderer itself is `pyre`.

## Current Capabilities

- Unidirectional path tracing with next-event estimation and MIS.
- Lambertian and Disney principled BSDFs, including GGX specular reflection.
- Triangle meshes, mesh instances, and a SAH-binned BVH.
- glTF loading with node transforms baked into mesh data.
- HDRI and procedural environment lighting with importance sampling.
- Thin-lens depth of field and translation-only motion blur.
- Progressive viewer window using `winit` and `pixels`.
- PNG output from linear-light film data.

## Workspace Layout

```text
crates/pyre/      Renderer library
crates/pyre-cli/  Command-line executable named pyre
docs/             Milestone images and supporting documentation
```

## Quick Start

Build the workspace:

```bash
cargo build
```

Render the default Cornell-box scene to `out.png`:

```bash
cargo run --bin pyre
```

Render a higher-resolution image with the optimized profile:

```bash
cargo run --release --bin pyre -- --output out.png --width 1920 --height 1080 --spp 64
```

Open the progressive viewer:

```bash
cargo run --bin pyre -- --viewer --spp 0
```

Load a glTF scene:

```bash
cargo run --release --bin pyre -- --scene path/to/scene.glb --output scene.png --spp 64
```

Use an HDRI environment:

```bash
cargo run --release --bin pyre -- --preset studio --env path/to/studio.hdr --env-intensity 1.0
```

## CLI Options

The `pyre` binary supports:

- `--output <PATH>`: output PNG path.
- `--scene <PATH>`: glTF or GLB input scene.
- `--width <N>` and `--height <N>`: render resolution.
- `--spp <N>`: samples per pixel. In viewer mode, `0` renders until closed.
- `--max-depth <N>`: maximum path depth.
- `--viewer`: open the progressive preview window.
- `--preset cornell|studio`: built-in scene preset.
- `--env <PATH>` and `--env-intensity <F>`: Radiance `.hdr` environment map.
- `--aperture <F>` and `--focus-distance <F>`: thin-lens camera controls.
- `--motion-blur`: enable built-in scene motion blur.

Run `cargo run --bin pyre -- --help` for the full generated help text.

## Development Commands

```bash
cargo build
cargo build --release
cargo test
cargo test -p pyre <test-name>
cargo clippy --workspace --all-targets
cargo fmt --all
```

The development profile intentionally keeps project code at `opt-level = 1` and
dependencies at `opt-level = 3`. This keeps compile times reasonable while
preserving usable SIMD performance for math-heavy debug renders.

## Architecture

Pyre is organized around renderer subsystems:

- `math`: rays, bounds, and orthonormal frames built on `glam`.
- `geometry`: shapes, triangle meshes, mesh instances, motion, and BVH.
- `camera`: pinhole and thin-lens camera implementations.
- `sampler`: deterministic per-pixel sampling.
- `material`: BSDF traits and material implementations.
- `light`: area lights and environment lights.
- `scene`: primitives, materials, lights, and intersection queries.
- `integrator`: path tracing and MIS logic.
- `film`: render buffers and image output.
- `io`: scene and image loading.
- `viewer`: optional progressive preview window.

The code is CPU-first today. The trait boundaries are intended to keep room for
future backends without forcing GPU concerns into the current implementation.

## Roadmap

Completed milestones include sphere normals, triangle meshes with BVH, path
tracing with MIS, Disney BSDF, progressive viewing, HDRI environments, depth of
field, and motion blur.

Upcoming work includes PBRT parsing, AOV and EXR output, denoising, USD
ingestion, and a GPU backend.

## License

This project is licensed under either of:

- MIT License
- Apache License, Version 2.0
