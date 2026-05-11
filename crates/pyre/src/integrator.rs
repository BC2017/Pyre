//! Light transport integrators. Milestone 3 ships only the unidirectional
//! path tracer with multiple importance sampling for direct lighting.

use crate::{
    math::{Frame, Ray},
    sampler::Sampler,
    scene::{HitKind, Scene},
};
use glam::Vec3;

/// Bias amount when spawning rays off a surface — pushes the origin off the
/// surface to dodge self-intersection. Scaled in scene units; for a Cornell
/// box of half-extent 1, 1e-3 is plenty.
const RAY_EPS: f32 = 1e-3;

pub struct PathIntegrator {
    pub max_depth: u32,
    pub min_rr_depth: u32,
}

impl Default for PathIntegrator {
    fn default() -> Self {
        Self {
            max_depth: 8,
            min_rr_depth: 3,
        }
    }
}

impl PathIntegrator {
    /// Estimate the radiance arriving at the camera along `ray` using a single
    /// path sample. The integrator uses MIS direct lighting (light + BSDF
    /// sampling combined via the power heuristic) and Russian roulette
    /// termination after `min_rr_depth` bounces.
    pub fn li(&self, mut ray: Ray, scene: &Scene, sampler: &mut impl Sampler) -> Vec3 {
        let mut l = Vec3::ZERO;
        let mut beta = Vec3::ONE;
        // pdf of the BSDF sample that produced the current ray. Used for MIS
        // when that ray happens to hit a light.
        let mut last_pdf_bsdf: f32 = 0.0;
        // Camera rays and specular bounces collect Le without MIS weighting.
        let mut last_was_specular = true;

        for depth in 0..self.max_depth {
            let Some(hit) = scene.intersect(&ray) else {
                // Ray escaped — pick up environment radiance if any. MIS
                // with the BSDF arm of NEE so the env contribution is
                // unbiased even when sampled from both arms.
                if let Some(env) = scene.env.as_ref() {
                    let le = env.le(ray.direction);
                    if le != Vec3::ZERO {
                        if last_was_specular {
                            l += beta * le;
                        } else {
                            let pdf_env = env.pdf(ray.direction);
                            let w_bsdf = power_heuristic(last_pdf_bsdf, pdf_env);
                            l += beta * le * w_bsdf;
                        }
                    }
                }
                break;
            };

            match hit.kind {
                HitKind::Light { light_id } => {
                    let light = &scene.lights[light_id as usize];
                    let le = light.le(
                        hit.interaction.position,
                        hit.interaction.normal,
                        -ray.direction,
                    );
                    if last_was_specular {
                        l += beta * le;
                    } else {
                        // BSDF→light arm of MIS direct lighting.
                        let pdf_light = light.pdf(ray.origin, ray.direction);
                        let w_bsdf = power_heuristic(last_pdf_bsdf, pdf_light);
                        l += beta * le * w_bsdf;
                    }
                    break;
                }
                HitKind::Surface { material_id, .. } => {
                    let material = &scene.materials[material_id as usize];
                    let p = hit.interaction.position;
                    let ns = hit.interaction.normal;
                    let frame = Frame::from_normal(ns);
                    let wo_local = frame.to_local(-ray.direction);
                    if wo_local.z <= 0.0 {
                        // Hit a back face — no transmission BSDFs yet.
                        break;
                    }

                    // Direct lighting (NEE) with MIS — light-sample arm.
                    for light in &scene.lights {
                        let Some(ls) = light.sample(p, sampler.next_vec2()) else {
                            continue;
                        };
                        if ls.pdf <= 0.0 || ls.li == Vec3::ZERO {
                            continue;
                        }
                        let wi_local = frame.to_local(ls.wi);
                        if wi_local.z <= 0.0 {
                            continue;
                        }
                        let f = material.eval(wo_local, wi_local);
                        if f == Vec3::ZERO {
                            continue;
                        }
                        // Visibility — bias both ends to skip the surface itself
                        // and the light geometry.
                        let p_offset = p + RAY_EPS * ns;
                        let target = ls.position - RAY_EPS * ls.wi;
                        if scene.occluded(p_offset, target, ray.time) {
                            continue;
                        }
                        let pdf_bsdf = material.pdf(wo_local, wi_local);
                        let w_light = power_heuristic(ls.pdf, pdf_bsdf);
                        l += beta * f * ls.li * wi_local.z * w_light / ls.pdf;
                    }

                    // Direct lighting (NEE) — extra arm for the environment.
                    // Environment lights are at infinity, so we shoot the
                    // shadow ray as a directional probe (`occluded_dir`).
                    if let Some(env) = scene.env.as_ref() {
                        let es = env.sample(sampler.next_vec2());
                        if es.pdf > 0.0 && es.li != Vec3::ZERO {
                            let wi_local = frame.to_local(es.wi);
                            if wi_local.z > 0.0 {
                                let f = material.eval(wo_local, wi_local);
                                if f != Vec3::ZERO {
                                    let p_offset = p + RAY_EPS * ns;
                                    if !scene.occluded_dir(p_offset, es.wi, ray.time) {
                                        let pdf_bsdf = material.pdf(wo_local, wi_local);
                                        let w_light = power_heuristic(es.pdf, pdf_bsdf);
                                        l += beta * f * es.li * wi_local.z * w_light / es.pdf;
                                    }
                                }
                            }
                        }
                    }

                    // BSDF sample — drives indirect lighting and the BSDF arm of MIS.
                    let Some(bs) = material.sample(wo_local, sampler.next_vec2()) else {
                        break;
                    };
                    if bs.pdf <= 0.0 {
                        break;
                    }
                    let cos_wi = bs.wi.z.abs();
                    if cos_wi == 0.0 {
                        break;
                    }
                    beta *= bs.f * cos_wi / bs.pdf;

                    if depth >= self.min_rr_depth {
                        let q = (1.0 - beta.max_element()).clamp(0.05, 0.95);
                        if sampler.next_f32() < q {
                            break;
                        }
                        beta /= 1.0 - q;
                    }

                    last_pdf_bsdf = bs.pdf;
                    last_was_specular = false;

                    let wi_world = frame.to_world(bs.wi);
                    ray = Ray {
                        origin: p + RAY_EPS * ns,
                        direction: wi_world,
                        t_min: 1e-4,
                        t_max: f32::INFINITY,
                        time: ray.time,
                    };
                }
            }
        }
        l
    }
}

#[inline]
fn power_heuristic(pa: f32, pb: f32) -> f32 {
    let a2 = pa * pa;
    let b2 = pb * pb;
    let denom = a2 + b2;
    if denom > 0.0 { a2 / denom } else { 0.0 }
}
