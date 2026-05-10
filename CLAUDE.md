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

- **math** — `Ray`, `Bounds3`, `Frame` (orthonormal basis around a normal). `glam` is the linalg primitive; treat it as a re-exported foundation rather than wrapping it in newtypes.
- **color** *(planned)* — RGB, spectral, tonemapping. Spectral support comes after the core integrator works.
- **geometry** — `Shape` trait, `Sphere`, `TriangleMesh`, `MeshInstance` (mesh + BVH), and our own SAH-binned `Bvh`. The `Shape` trait is what lights, integrators, and scene queries see.
- **scene** — `Scene { primitives, materials, lights }`, `Primitive` (mesh instance + material id), `SceneHit` enum tagged Surface/Light. Linear iteration today; a top-level BVH (TLAS) lands at milestone 5 if perf needs it.
- **camera** — `Camera` trait (`PinholeCamera` exists; `ThinLensCamera` next).
- **sampler** — `Sampler` trait + `IndependentSampler` (xoshiro256++). Per-pixel deterministic seeds via `pixel_seed(x, y, sample)`. Stratified / Halton / Sobol come later.
- **material** — `Bsdf` trait + `Lambertian` + `DisneyBsdf` (Disney diffuse + GGX specular reflection, parameters: base_color/metallic/roughness/specular). Transmission, sheen, clearcoat, and anisotropic are deferred to a milestone 4-followup.
- **light** — `Light` trait + `DiffuseAreaQuadLight`. Lights live separately from primitives; shadow rays test only primitives.
- **integrator** — `PathIntegrator`: unidirectional path tracer with MIS direct lighting (NEE light-sample arm + BSDF-sample arm via the power heuristic) and Russian roulette.
- **film** — Tile management, AOVs, PNG/EXR writers.
- **io** — `load_gltf` (walks the default scene's node hierarchy, bakes transforms into vertex data). PBRT loader and a `SceneLoader` trait come with milestone 7.
- **viewer** — winit + pixels progressive preview window. Gated by the `viewer` Cargo feature on `pyre`. A render thread accumulates samples into a shared running-mean buffer (Welford); the event loop redraws on each completed pass. `S` saves a snapshot, `Esc`/`Q` quits, target spp triggers auto-save+exit.

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
| 2 ✅ | Triangle meshes + BVH + glTF loader | geometry::TriangleMesh, MeshInstance, Bvh, io::gltf |
| 3 ✅ | Path tracing with MIS | sampler, material, light, scene, integrator |
| 4 ✅ | Disney principled BSDF (diffuse + GGX specular) | material::DisneyBsdf |
| 5 ✅ | Progressive viewer window | viewer (winit + pixels) |
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

## Notes by milestone

- **2** — `MeshInstance` (mesh + BVH) is the renderable unit; bare `TriangleMesh` is just data. The BVH is single-level (BLAS over triangles); a top-level BVH over instances arrives with the `scene` module. The CLI keeps a flat `World { instances: Vec<MeshInstance> }` and iterates linearly — this is fine for one or two meshes but should be replaced before scenes get large. Tests in `geometry::mesh::tests` cover triangle hits, misses, and BVH closest-hit selection.
- **3** — Path tracer is the textbook "MIS direct lighting + cosine-weighted BSDF sampling + Russian roulette" recipe. Lights are stored on `Scene` separately from primitives (so shadow rays don't have to skip emitters); when a BSDF-sampled ray happens to land on a light, the integrator recognises it via `SceneHit::Light` and applies the power-heuristic MIS weight. Camera rays and (eventually) specular bounces collect Le without MIS — that's the `last_was_specular` flag in `integrator::PathIntegrator`. The `Scene::bounds()` used for auto-framing intentionally excludes lights, so distant emitters won't pull the camera back. Cargo workspace bumped from `rand 0.8` to `rand 0.9` (and `rand_xoshiro 0.6 → 0.7`) to dodge Rust 2024's reserved `gen` keyword.
- **4** — `DisneyBsdf` is the "one-sample model" multi-lobe BSDF: pick a lobe by weight, sample within it, but evaluate `f` and `pdf` as the sum/weighted-sum across all lobes — this gives unbiased MIS in the path integrator. Specular uses Heitz 2018 VNDF sampling for GGX (better variance than D-only sampling at grazing angles). `material::tests::disney_sample_eval_pdf_consistency` is load-bearing: it runs `sample → eval → pdf` round-trips across 1500 random configs and is the first thing to verify when adding lobes, since BSDF errors manifest as quiet bias rather than crashes. `Lambertian` is kept around — it's strictly simpler and still useful for matte walls.
- **5** — Viewer is its own Cargo feature on `pyre` (`viewer = ["dep:winit", "dep:pixels", "dep:anyhow"]`) so library consumers that only want the offline renderer don't pull wgpu. `pyre-cli` enables it. Architecture is a render thread (rayon over rows) producing one full SPP per pass into a scratch buffer, then merging into a `Mutex<Vec<Vec3>>` accumulator with the Welford running-mean update; the winit `ApplicationHandler` event loop reads the accumulator under lock on `RedrawRequested` and presents through `pixels::Pixels`. Communication is via `EventLoopProxy<UserEvent>` (`PassCompleted` requests a redraw + logs; `RenderFinished` triggers auto-save + exit). The `Pixels<'static>` lifetime is satisfied by leaking the `Window` once with `Box::leak` — bounded leak per CLI invocation. `--viewer --spp 0` runs unbounded; non-zero spp auto-saves on completion, which is also what enables scripted reference renders to use the viewer code path end-to-end. No interactive camera, resize, or zoom — those are deferred.
