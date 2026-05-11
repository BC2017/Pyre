//! Progressive preview window. A render thread accumulates samples into a
//! shared per-pixel running mean; the winit event loop reads the latest
//! mean after each completed pass, gamma-encodes, and presents through a
//! `pixels` framebuffer.
//!
//! Behaviour is intentionally minimal for milestone 5: the window is
//! one-to-one with the render resolution, no resize, no interactive
//! camera, no zoom. Keys:
//!
//! - `S` saves a PNG snapshot of the current accumulator to the path in
//!   `ViewerConfig::snapshot_path`.
//! - `Esc` or `Q` aborts the render and closes the window without saving.
//! - When `target_spp` is set and reached, the snapshot is auto-saved and
//!   the window closes (so scripted reference renders work end-to-end).
//!
//! The `viewer` Cargo feature must be enabled to compile this module.

use crate::{
    Camera, CameraSample, Film, IndependentSampler, PathIntegrator, Sampler, Scene, pixel_seed,
};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Debug, Clone)]
pub struct ViewerConfig {
    pub width: u32,
    pub height: u32,
    /// Stop after this many passes. `None` runs until the user closes the
    /// window. When `Some(n)` is reached, the snapshot is auto-saved.
    pub target_spp: Option<u32>,
    /// Where the `S` key (and the auto-save on completion) writes the PNG.
    pub snapshot_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
enum UserEvent {
    PassCompleted,
    RenderFinished,
}

struct RenderState {
    width: u32,
    height: u32,
    /// Running per-pixel mean of converged radiance. Welford-style update
    /// in `render_loop` so the buffer is always presentable.
    accum: Mutex<Vec<Vec3>>,
    /// Number of completed passes (= samples per pixel). Loaded by the
    /// event loop to know whether to draw and what to log.
    pass_count: AtomicU32,
    /// Set by the event loop when the user requests a quit; the render
    /// thread polls between passes and exits.
    stop: AtomicBool,
    target_spp: Option<u32>,
}

pub fn run(
    scene: Scene,
    camera: Box<dyn Camera>,
    integrator: PathIntegrator,
    config: ViewerConfig,
) -> anyhow::Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    let state = Arc::new(RenderState {
        width: config.width,
        height: config.height,
        accum: Mutex::new(vec![
            Vec3::ZERO;
            (config.width as usize) * (config.height as usize)
        ]),
        pass_count: AtomicU32::new(0),
        stop: AtomicBool::new(false),
        target_spp: config.target_spp,
    });

    let render_thread = {
        let state = Arc::clone(&state);
        let scene = Arc::new(scene);
        let proxy = proxy.clone();
        std::thread::Builder::new()
            .name("pyre-render".into())
            .spawn(move || render_loop(state, scene, camera, integrator, proxy))?
    };

    let mut app = ViewerApp {
        state: Arc::clone(&state),
        config,
        window: None,
        pixels: None,
        start: Instant::now(),
        last_logged_pass: 0,
    };
    let result = event_loop.run_app(&mut app).map_err(anyhow::Error::from);

    // Whatever happened in the event loop, signal the render thread to wind
    // down before we return — otherwise it would block process exit until
    // the next pass finishes.
    state.stop.store(true, Ordering::Relaxed);
    let _ = render_thread.join();
    result
}

fn render_loop(
    state: Arc<RenderState>,
    scene: Arc<Scene>,
    camera: Box<dyn Camera>,
    integrator: PathIntegrator,
    proxy: EventLoopProxy<UserEvent>,
) {
    let width = state.width;
    let height = state.height;
    // Persistent scratch buffer reused across passes — avoids re-allocating
    // ~width*height*Vec3 every pass, which is significant for HD frames.
    let mut pass_buffer: Vec<Vec3> =
        vec![Vec3::ZERO; (width as usize) * (height as usize)];
    let mut pass: u32 = 0;

    loop {
        if state.stop.load(Ordering::Relaxed) {
            return;
        }
        if let Some(target) = state.target_spp {
            if pass >= target {
                let _ = proxy.send_event(UserEvent::RenderFinished);
                return;
            }
        }

        // One full sample-per-pixel pass.
        pass_buffer
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let y = y as u32;
                for (x, pixel) in row.iter_mut().enumerate() {
                    let x = x as u32;
                    let mut sampler = IndependentSampler::new(pixel_seed(x, y, pass));
                    let jitter_x = sampler.next_f32();
                    let jitter_y = sampler.next_f32();
                    let ndc_x = 2.0 * (x as f32 + jitter_x) / width as f32 - 1.0;
                    let ndc_y = 1.0 - 2.0 * (y as f32 + jitter_y) / height as f32;
                    let lens = sampler.next_vec2();
                    let time = sampler.next_f32();
                    let ray = camera.generate_ray(CameraSample {
                        ndc: Vec2::new(ndc_x, ndc_y),
                        lens,
                        time,
                    });
                    *pixel = integrator.li(ray, &scene, &mut sampler);
                }
            });
        pass += 1;

        // Merge the pass into the running mean. Welford update so the
        // visible buffer is always the displayable mean — no divide on read.
        {
            let mut accum = state.accum.lock().unwrap();
            let n = pass as f32;
            accum
                .par_iter_mut()
                .zip(pass_buffer.par_iter())
                .for_each(|(d, &s)| {
                    *d += (s - *d) / n;
                });
        }
        state.pass_count.store(pass, Ordering::Relaxed);

        if proxy.send_event(UserEvent::PassCompleted).is_err() {
            // Event loop closed — bail.
            return;
        }
    }
}

struct ViewerApp {
    state: Arc<RenderState>,
    config: ViewerConfig,
    /// Held as `&'static` via `Box::leak`. The viewer runs once per CLI
    /// invocation and the OS reclaims on process exit, so a single bounded
    /// leak is the cheapest way to satisfy `pixels::Pixels<'static>`.
    window: Option<&'static Window>,
    pixels: Option<pixels::Pixels<'static>>,
    start: Instant,
    last_logged_pass: u32,
}

impl ApplicationHandler<UserEvent> for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Pyre — viewer")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ))
            .with_resizable(false);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                tracing::error!(error = ?e, "create_window failed");
                event_loop.exit();
                return;
            }
        };
        let window: &'static Window = Box::leak(Box::new(window));
        let surface =
            pixels::SurfaceTexture::new(self.config.width, self.config.height, window);
        let pixels = match pixels::Pixels::new(self.config.width, self.config.height, surface) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(error = ?e, "pixels::Pixels::new failed");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window);
        self.pixels = Some(pixels);
        self.start = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.state.stop.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match code {
                KeyCode::Escape | KeyCode::KeyQ => {
                    self.state.stop.store(true, Ordering::Relaxed);
                    event_loop.exit();
                }
                KeyCode::KeyS => self.save_snapshot(),
                _ => {}
            },
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::PassCompleted => {
                let pass = self.state.pass_count.load(Ordering::Relaxed);
                if pass != self.last_logged_pass {
                    let elapsed = self.start.elapsed();
                    tracing::info!(
                        pass,
                        elapsed_ms = elapsed.as_millis() as u64,
                        "pass complete"
                    );
                    self.last_logged_pass = pass;
                }
                if let Some(window) = self.window {
                    window.request_redraw();
                }
            }
            UserEvent::RenderFinished => {
                self.save_snapshot();
                event_loop.exit();
            }
        }
    }
}

impl ViewerApp {
    fn draw(&mut self) {
        let Some(pixels) = self.pixels.as_mut() else {
            return;
        };
        let pass = self.state.pass_count.load(Ordering::Relaxed);
        let frame = pixels.frame_mut();
        if pass == 0 {
            for chunk in frame.chunks_exact_mut(4) {
                chunk.copy_from_slice(&[0, 0, 0, 0xff]);
            }
        } else {
            let accum = self.state.accum.lock().unwrap();
            for (i, chunk) in frame.chunks_exact_mut(4).enumerate() {
                let p = accum[i];
                let encoded = p.clamp(Vec3::ZERO, Vec3::ONE).powf(1.0 / 2.2) * 255.0;
                chunk[0] = encoded.x as u8;
                chunk[1] = encoded.y as u8;
                chunk[2] = encoded.z as u8;
                chunk[3] = 0xff;
            }
        }
        if let Err(e) = pixels.render() {
            tracing::error!(error = ?e, "pixels::render failed");
        }
    }

    fn save_snapshot(&self) {
        let pass = self.state.pass_count.load(Ordering::Relaxed);
        if pass == 0 {
            tracing::warn!("no completed passes — snapshot skipped");
            return;
        }
        let snapshot = {
            let accum = self.state.accum.lock().unwrap();
            accum.clone()
        };
        let film = Film::from_buffer(self.config.width, self.config.height, snapshot);
        match film.save_png(&self.config.snapshot_path) {
            Ok(()) => tracing::info!(
                pass,
                path = %self.config.snapshot_path.display(),
                "snapshot saved"
            ),
            Err(e) => tracing::error!(
                error = ?e,
                path = %self.config.snapshot_path.display(),
                "snapshot save failed"
            ),
        }
    }
}
