//! Per-pixel sample sources. The integrator pulls 1D and 2D uniform samples
//! through the `Sampler` trait; the trait exists so future sessions can
//! drop in stratified / Halton / Sobol implementations without touching
//! the integrator.

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub trait Sampler: Send {
    fn next_f32(&mut self) -> f32;
    fn next_vec2(&mut self) -> Vec2;
}

#[derive(Clone)]
pub struct IndependentSampler {
    rng: Xoshiro256PlusPlus,
}

impl IndependentSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

impl Sampler for IndependentSampler {
    fn next_f32(&mut self) -> f32 {
        self.rng.random::<f32>()
    }

    fn next_vec2(&mut self) -> Vec2 {
        Vec2::new(self.rng.random::<f32>(), self.rng.random::<f32>())
    }
}

/// Hash (pixel x, pixel y, sample index) into a 64-bit seed via SplitMix64.
/// Deterministic across runs and platforms — important for reproducible
/// regression renders.
pub fn pixel_seed(x: u32, y: u32, sample: u32) -> u64 {
    let mut h = ((x as u64) << 32) | (y as u64);
    h ^= (sample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}
