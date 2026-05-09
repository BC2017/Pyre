# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pyre is a high-fidelity offline path tracer (V-Ray / Arnold style). CPU-first with a trait-based architecture so a GPU backend can slot in later. Aimed at a serious portfolio project — production-quality algorithms (path tracing with MIS, Disney BSDF, importance sampling) but pure-Rust toolchain where feasible.

The workspace is named `rust-engine` for historical reasons; the renderer itself is `pyre`.

## Commands

```
cargo build                          # Build the workspace
cargo build --release                # Optimized build — use this for any non-trivial render
cargo run --bin pyre                 # Render with defaults to out.png (dev profile is fast enough)
cargo run --release --bin pyre -- --output out.png --width 1920 --height 1080
cargo test                           # All tests
cargo test -p pyre <name>            # Single test in the engine crate
cargo clippy --workspace --all-targets
cargo fmt --all
```

The dev profile is set to `opt-level = 1` with dependencies at `opt-level = 3`. Without that, glam SIMD math is slow enough that `cargo run` renders are unusable. Don't lower these.

## Workspace layout

- `crates/pyre/` — the renderer library
- `crates/pyre-cli/` — binary `pyre` (command-line entry point)
- `scenes/` — test scenes (Cornell box, etc.)

## Architecture

CPU-first unidirectional path tracer. The integrator loop is parallelized per pixel/tile via rayon. The library is organized as one module per responsibility, each typically exposing a trait + implementations:

- **math** — `Ray`, `Bounds3`. `glam` is the linalg primitive; treat it as a re-exported foundation rather than wrapping it in newtypes.
- **color** *(planned)* — RGB, spectral, tonemapping. Spectral support comes after the core integrator works.
- **geometry** — `Shape` trait (`Sphere`, `TriangleMesh`), our own BVH (not embree). The `Shape` trait is what lights, integrators, and scene queries see.
- **scene** *(planned)* — scene graph, instancing, transforms.
- **camera** — `Camera` trait (`PinholeCamera` exists; `ThinLensCamera` next).
- **sampler** *(planned)* — `Sampler` trait (`Halton`, `Sobol`, stratified). Per-pixel deterministic state — never `rand::thread_rng()` in hot paths.
- **material** *(planned)* — `BSDF` trait, Disney principled BSDF as the default.
- **light** *(planned)* — `Light` trait (area, point, distant, HDRI environment).
- **integrator** *(planned)* — `Integrator` trait, unidirectional path tracer with MIS.
- **film** — Tile management, AOVs, PNG/EXR writers.
- **io** *(planned)* — `SceneLoader` trait. glTF + PBRT first; USD deferred.
- **viewer** *(planned)* — winit + pixels progressive preview window.

Modules marked *(planned)* don't exist yet — add them when the corresponding milestone lands.

## Conventions

- **Math:** linear algebra via `glam`. Use `Vec3` for points/directions/colors; introduce newtypes only when ambiguity actually bites in a specific module.
- **Coordinate system:** right-handed, +Y up, +Z toward viewer (matches glTF).
- **Errors:** `thiserror` in the library, `anyhow` in the CLI.
- **Logging:** all via `tracing` macros. The CLI installs `tracing_subscriber::fmt` with `RUST_LOG` env filter.
- **Determinism:** all randomness goes through the `Sampler` trait so renders are reproducible. No `thread_rng()` inside the integrator or its callees.
- **Image data:** the `Film` stores linear-light radiance. Gamma encoding happens only at the PNG/JPG sink. EXR (when added) writes linear values verbatim.
- **Parallelism:** `rayon`'s `par_chunks_mut` over rows or tiles. Anything reachable from the integrator must be `Send + Sync`.

## Roadmap

The renderer should always be runnable. Each milestone adds a visible capability without breaking the previous one.

| # | Milestone | Adds |
|---|---|---|
| 1 ✅ | Normals on a sphere | math, geometry::Sphere, camera, film, PNG output |
| 2 | Triangle meshes + BVH + glTF loader | geometry::TriangleMesh, BVH, io::gltf |
| 3 | Path tracing with MIS | sampler, integrator, light (area), Lambertian BRDF |
| 4 | Disney principled BSDF | material |
| 5 | Progressive viewer window | viewer (winit + pixels) |
| 6 | HDRI envs, thin-lens DoF, motion blur | light::env, camera::ThinLens, time-varying transforms |
| 7 | PBRT scene parser | io::pbrt |
| 8 | AOVs + EXR | film AOV channels, exr writer |
| 9 | OIDN denoising | feature-gated `oidn` integration |
| 10 | USD ingestion | cxx FFI to OpenUSD |
| 11 | GPU backend | wgpu compute or CUDA via `cust`, behind existing traits |

When closing a milestone, tick the box in this table and add a one-line summary to the bottom of this file noting any deviations from the plan.

## Deferred dependencies

These are intentionally **not** in `Cargo.toml`. Don't add them without revisiting the plan:

- **`embree4`** — Intel Embree CPU ray-tracing kernels. Roll our own BVH first. Embree only if profiling shows traversal as the bottleneck.
- **`oidn`** — Intel Open Image Denoise. Adds a C++ build dep; not worth gating MVP on.
- **OpenUSD via `cxx`** — milestone 10. Real C++ build engineering; will bring its own README section when added.

When a milestone needs a workspace-listed dep (e.g., `gltf` at milestone 2), add it to that crate's `Cargo.toml` with `gltf.workspace = true`. Don't restate version numbers — the workspace owns those.

## Modifying this file

If you add a new top-level module to the engine, change the integrator architecture, or alter the build profile, update this file in the same change. CLAUDE.md is the contract for future sessions; drift here costs other people time.
