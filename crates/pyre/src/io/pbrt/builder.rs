//! Translates parsed PBRT directives into a `Scene` plus optional render
//! settings (camera placement, output resolution). Maintains a transform
//! stack and a material/area-light state stack so `AttributeBegin` /
//! `AttributeEnd` blocks behave as in the PBRT spec.
//!
//! What's supported is intentionally a minimum subset (see `mod.rs` for
//! the directive list). Unsupported directives are logged via `tracing`
//! and skipped rather than failing the whole load — partial loads are
//! useful while bringing new shape/material types up.
//!
//! Coordinate convention: pyre uses right-handed +Y up. PBRT's spec is
//! +Z up by default, so PBRT files written for other renderers may load
//! upside-down. The hand-authored `scenes/cornell.pbrt` we ship is
//! written for +Y up to round-trip the built-in Cornell box exactly.

use super::parser::{Directive, Param, ParamValues, PositionalArg};
use crate::{
    Bsdf, DiffuseAreaQuadLight, DisneyBsdf, HdriEnvironmentLight, Lambertian, MeshInstance,
    Primitive, Scene, TriangleMesh, load_hdri,
};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct LoadedPbrt {
    pub scene: Scene,
    pub camera: Option<CameraSpec>,
    pub film: Option<FilmSpec>,
}

#[derive(Debug, Clone, Copy)]
pub struct CameraSpec {
    pub origin: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub vfov_deg: f32,
}

#[derive(Debug, Clone)]
pub struct FilmSpec {
    pub width: u32,
    pub height: u32,
    pub filename: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("AttributeEnd without matching AttributeBegin (line {line})")]
    UnbalancedAttribute { line: u32 },
    #[error("TransformEnd without matching TransformBegin (line {line})")]
    UnbalancedTransform { line: u32 },
    #[error("named material {0:?} not declared")]
    UnknownNamedMaterial(String),
    #[error("HDRI load failed: {0}")]
    HdrLoad(#[from] crate::io::HdrLoadError),
    #[error("Shape \"{0}\" requires parameter {1:?}")]
    MissingShapeParam(String, String),
    #[error("malformed {0} on line {1}: {2}")]
    Malformed(String, u32, String),
}

#[derive(Clone)]
struct Attribute {
    ctm: Mat4,
    material_id: Option<u32>,
    area_light: Option<Vec3>,
}

pub fn build(directives: Vec<Directive>, base_dir: &Path) -> Result<LoadedPbrt, BuildError> {
    let mut scene = Scene::new();
    let mut named_materials: HashMap<String, u32> = HashMap::new();

    let mut attr_stack: Vec<Attribute> = Vec::new();
    let mut transform_stack: Vec<Mat4> = Vec::new();

    let mut cur = Attribute {
        ctm: Mat4::IDENTITY,
        material_id: None,
        area_light: None,
    };

    // LookAt sets a "camera transform" that PBRT applies on top of the
    // current CTM at Camera-creation time. We track it separately so the
    // CTM stack is reserved for object transforms.
    let mut lookat: Option<(Vec3, Vec3, Vec3)> = None;
    let mut camera_vfov: Option<f32> = None;
    let mut film: Option<FilmSpec> = None;

    let mut in_world = false;

    for d in directives {
        match d.keyword.as_str() {
            // ---------------- pre-world: render settings ----------------
            "LookAt" => {
                let f = expect_floats(&d, 9, "LookAt")?;
                lookat = Some((
                    Vec3::new(f[0], f[1], f[2]),
                    Vec3::new(f[3], f[4], f[5]),
                    Vec3::new(f[6], f[7], f[8]),
                ));
            }
            "Camera" => {
                let name = expect_first_string(&d, "Camera")?;
                if name != "perspective" {
                    tracing::warn!(
                        camera = %name,
                        "only \"perspective\" Camera supported; ignoring others"
                    );
                }
                camera_vfov = d
                    .params
                    .iter()
                    .find(|p| p.name == "fov")
                    .and_then(|p| first_float(p))
                    .map(|v| v as f32);
            }
            "Film" => {
                let width = d
                    .params
                    .iter()
                    .find(|p| p.name == "xresolution")
                    .and_then(|p| first_int(p))
                    .unwrap_or(800) as u32;
                let height = d
                    .params
                    .iter()
                    .find(|p| p.name == "yresolution")
                    .and_then(|p| first_int(p))
                    .unwrap_or(600) as u32;
                let filename = d
                    .params
                    .iter()
                    .find(|p| p.name == "filename")
                    .and_then(|p| p.as_strings())
                    .and_then(|v| v.first())
                    .map(PathBuf::from);
                film = Some(FilmSpec {
                    width,
                    height,
                    filename,
                });
            }
            "Sampler" | "Integrator" | "Accelerator" | "PixelFilter" => {
                // Pyre uses its own sampler/integrator/accelerator. Parsing
                // these gives forward compatibility with full PBRT files but
                // they don't change rendering.
            }

            // ---------------- transforms ----------------
            "Identity" => cur.ctm = Mat4::IDENTITY,
            "Translate" => {
                let f = expect_floats(&d, 3, "Translate")?;
                cur.ctm *= Mat4::from_translation(Vec3::new(f[0], f[1], f[2]));
            }
            "Scale" => {
                let f = expect_floats(&d, 3, "Scale")?;
                cur.ctm *= Mat4::from_scale(Vec3::new(f[0], f[1], f[2]));
            }
            "Rotate" => {
                let f = expect_floats(&d, 4, "Rotate")?;
                let angle = (f[0] as f32).to_radians();
                let axis = Vec3::new(f[1], f[2], f[3]).normalize();
                cur.ctm *= Mat4::from_quat(Quat::from_axis_angle(axis, angle));
            }
            "Transform" | "ConcatTransform" => {
                let arr = match d.positional.first() {
                    Some(PositionalArg::FloatArray(a)) if a.len() == 16 => a.clone(),
                    _ => {
                        return Err(BuildError::Malformed(
                            d.keyword.clone(),
                            d.pos.line,
                            "expected a 16-float array".into(),
                        ));
                    }
                };
                let m = mat4_from_pbrt_row_major(&arr);
                cur.ctm = if d.keyword == "Transform" { m } else { cur.ctm * m };
            }

            // ---------------- block markers ----------------
            "AttributeBegin" => attr_stack.push(cur.clone()),
            "AttributeEnd" => {
                cur = attr_stack
                    .pop()
                    .ok_or(BuildError::UnbalancedAttribute { line: d.pos.line })?;
            }
            "TransformBegin" => transform_stack.push(cur.ctm),
            "TransformEnd" => {
                cur.ctm = transform_stack
                    .pop()
                    .ok_or(BuildError::UnbalancedTransform { line: d.pos.line })?;
            }
            "WorldBegin" => {
                in_world = true;
                cur.ctm = Mat4::IDENTITY;
            }
            "WorldEnd" => {
                in_world = false;
            }

            // ---------------- materials ----------------
            "Material" => {
                let name = expect_first_string(&d, "Material")?;
                let mat = build_material(&name, &d)?;
                let id = scene.materials.len() as u32;
                scene.materials.push(mat);
                cur.material_id = Some(id);
            }
            "MakeNamedMaterial" => {
                let name = expect_first_string(&d, "MakeNamedMaterial")?;
                let type_name = d
                    .params
                    .iter()
                    .find(|p| p.name == "type")
                    .and_then(|p| p.as_strings())
                    .and_then(|v| v.first())
                    .cloned()
                    .unwrap_or_else(|| "matte".to_string());
                let mat = build_material(&type_name, &d)?;
                let id = scene.materials.len() as u32;
                scene.materials.push(mat);
                named_materials.insert(name, id);
            }
            "NamedMaterial" => {
                let name = expect_first_string(&d, "NamedMaterial")?;
                let id = named_materials
                    .get(&name)
                    .copied()
                    .ok_or(BuildError::UnknownNamedMaterial(name))?;
                cur.material_id = Some(id);
            }

            // ---------------- lights ----------------
            "AreaLightSource" => {
                let kind = expect_first_string(&d, "AreaLightSource")?;
                if kind != "diffuse" {
                    tracing::warn!(area_light = %kind, "only \"diffuse\" area lights supported");
                }
                let l = param_rgb(&d, "L").unwrap_or(Vec3::splat(1.0));
                cur.area_light = Some(l);
            }
            "LightSource" => {
                let kind = expect_first_string(&d, "LightSource")?;
                if !in_world {
                    tracing::warn!(
                        light = %kind,
                        "LightSource outside WorldBegin — Pyre still places it, but PBRT spec forbids this"
                    );
                }
                add_light_source(&kind, &d, base_dir, &mut scene)?;
            }

            // ---------------- shapes ----------------
            "Shape" => {
                if !in_world {
                    tracing::warn!("Shape outside WorldBegin — ignoring");
                    continue;
                }
                let kind = expect_first_string(&d, "Shape")?;
                add_shape(&kind, &d, &cur, &mut scene)?;
            }

            // ---------------- ignored / not-yet-supported ----------------
            "Texture" | "ObjectBegin" | "ObjectEnd" | "ObjectInstance" | "Include"
            | "ReverseOrientation" | "ActiveTransform" | "CoordinateSystem"
            | "CoordSysTransform" | "MediumInterface" | "MakeNamedMedium" => {
                tracing::warn!(
                    keyword = %d.keyword,
                    line = d.pos.line,
                    "PBRT directive parsed but not yet implemented; skipping"
                );
            }

            other => {
                tracing::warn!(keyword = %other, "unknown PBRT directive; skipping");
            }
        }
    }

    let camera = if let (Some((origin, target, up)), Some(vfov)) = (lookat, camera_vfov) {
        Some(CameraSpec {
            origin,
            target,
            up,
            vfov_deg: vfov,
        })
    } else {
        None
    };

    Ok(LoadedPbrt {
        scene,
        camera,
        film,
    })
}

// ============================================================================
// Material construction
// ============================================================================

fn build_material(kind: &str, d: &Directive) -> Result<Box<dyn Bsdf>, BuildError> {
    match kind {
        "matte" => {
            let kd = param_rgb(d, "Kd").unwrap_or(Vec3::splat(0.5));
            Ok(Box::new(Lambertian { albedo: kd }))
        }
        "plastic" => {
            // Approximation: route plastic into Disney with metallic=0,
            // base_color=Kd, and pick up roughness if given. Ks affects
            // specular "intensity"; map it loosely to Disney's `specular`.
            let kd = param_rgb(d, "Kd").unwrap_or(Vec3::splat(0.5));
            let ks = param_rgb(d, "Ks").unwrap_or(Vec3::splat(0.5));
            let roughness = param_float(d, "roughness").unwrap_or(0.2);
            Ok(Box::new(DisneyBsdf {
                base_color: kd,
                metallic: 0.0,
                roughness,
                specular: (ks.x + ks.y + ks.z) / 3.0,
            }))
        }
        "metal" | "conductor" => {
            // Disney metallic. PBRT's `metal` uses eta/k for the Fresnel —
            // we approximate by using `Kr` if present, otherwise the
            // reflectance of common gold (close enough for the demo scene).
            let base = param_rgb(d, "Kr")
                .or_else(|| param_rgb(d, "reflectance"))
                .unwrap_or(Vec3::new(1.0, 0.766, 0.336));
            let roughness = param_float(d, "roughness").unwrap_or(0.1);
            Ok(Box::new(DisneyBsdf {
                base_color: base,
                metallic: 1.0,
                roughness,
                specular: 0.5,
            }))
        }
        other => {
            tracing::warn!(
                material = %other,
                "PBRT material {other:?} not supported; falling back to matte gray"
            );
            Ok(Box::new(Lambertian {
                albedo: Vec3::splat(0.5),
            }))
        }
    }
}

// ============================================================================
// Light construction
// ============================================================================

fn add_light_source(
    kind: &str,
    d: &Directive,
    base_dir: &Path,
    scene: &mut Scene,
) -> Result<(), BuildError> {
    match kind {
        "infinite" => {
            let intensity = param_float(d, "scale").unwrap_or(1.0);
            if let Some(mapname) = d
                .params
                .iter()
                .find(|p| p.name == "mapname")
                .and_then(|p| p.as_strings())
                .and_then(|v| v.first())
            {
                let path = base_dir.join(mapname);
                let env = load_hdri(&path, intensity)?;
                scene.env = Some(Box::new(env));
            } else {
                let l = param_rgb(d, "L").unwrap_or(Vec3::splat(1.0));
                scene.env = Some(Box::new(HdriEnvironmentLight::constant(l, intensity)));
            }
            Ok(())
        }
        other => {
            tracing::warn!(
                light = %other,
                "PBRT LightSource {other:?} not supported; skipping"
            );
            Ok(())
        }
    }
}

// ============================================================================
// Shape construction
// ============================================================================

fn add_shape(
    kind: &str,
    d: &Directive,
    cur: &Attribute,
    scene: &mut Scene,
) -> Result<(), BuildError> {
    match kind {
        "sphere" => {
            let radius = param_float(d, "radius").unwrap_or(1.0);
            // Tessellate to a triangle mesh; bake the CTM in so all
            // downstream code sees world-space vertices.
            let mesh = uv_sphere_mesh(radius, 48, 24, cur.ctm);
            push_primitive_or_quad_light(mesh, cur, scene);
            Ok(())
        }
        "trianglemesh" => {
            let positions = d
                .params
                .iter()
                .find(|p| p.name == "P")
                .and_then(|p| p.as_floats())
                .ok_or_else(|| {
                    BuildError::MissingShapeParam("trianglemesh".into(), "P".into())
                })?;
            let indices = d
                .params
                .iter()
                .find(|p| p.name == "indices")
                .and_then(|p| p.as_ints())
                .ok_or_else(|| {
                    BuildError::MissingShapeParam("trianglemesh".into(), "indices".into())
                })?;
            if positions.len() % 3 != 0 {
                return Err(BuildError::Malformed(
                    "trianglemesh".into(),
                    d.pos.line,
                    "P length not a multiple of 3".into(),
                ));
            }
            if indices.len() % 3 != 0 {
                return Err(BuildError::Malformed(
                    "trianglemesh".into(),
                    d.pos.line,
                    "indices length not a multiple of 3".into(),
                ));
            }
            let mut world_positions: Vec<Vec3> = positions
                .chunks_exact(3)
                .map(|c| {
                    cur.ctm
                        .transform_point3(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
                })
                .collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();

            // Optional normals.
            let normals = d
                .params
                .iter()
                .find(|p| p.name == "N")
                .and_then(|p| p.as_floats())
                .map(|n| {
                    n.chunks_exact(3)
                        .map(|c| {
                            cur.ctm
                                .transform_vector3(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
                                .normalize()
                        })
                        .collect::<Vec<_>>()
                });

            // Special-case: emissive quad — surface lights in Pyre are
            // `DiffuseAreaQuadLight`, not generic emissive meshes. When an
            // AttributeBegin block ends with `AreaLightSource` + a four-
            // vertex two-triangle quad, convert it to a quad light instead.
            if let Some(emission) = cur.area_light {
                if world_positions.len() == 4 && indices_u32.len() == 6 {
                    let q = quad_from_corners(&world_positions);
                    scene.lights.push(Box::new(DiffuseAreaQuadLight::new(
                        q.p0,
                        q.edge_u,
                        q.edge_v,
                        emission,
                    )));
                    return Ok(());
                } else {
                    tracing::warn!(
                        line = d.pos.line,
                        "AreaLightSource attached to non-quad trianglemesh; skipping emission"
                    );
                    // Fall through and add as a non-emissive primitive.
                    let _ = &mut world_positions;
                }
            }

            let mesh = TriangleMesh {
                positions: world_positions,
                indices: indices_u32,
                normals,
                uvs: None,
            };
            let mat = cur.material_id.unwrap_or_else(|| {
                tracing::warn!(line = d.pos.line, "Shape with no current Material — using gray");
                push_default_material(scene)
            });
            scene.primitives.push(Primitive {
                instance: MeshInstance::build(mesh),
                material_id: mat,
            });
            Ok(())
        }
        other => {
            tracing::warn!(
                shape = %other,
                "PBRT Shape {other:?} not supported; skipping"
            );
            Ok(())
        }
    }
}

fn push_primitive_or_quad_light(mesh: TriangleMesh, cur: &Attribute, scene: &mut Scene) {
    if cur.area_light.is_some() {
        tracing::warn!("AreaLightSource attached to a non-quad shape (sphere?) — emission ignored");
    }
    let mat = cur.material_id.unwrap_or_else(|| push_default_material(scene));
    scene.primitives.push(Primitive {
        instance: MeshInstance::build(mesh),
        material_id: mat,
    });
}

fn push_default_material(scene: &mut Scene) -> u32 {
    let id = scene.materials.len() as u32;
    scene.materials.push(Box::new(Lambertian {
        albedo: Vec3::splat(0.5),
    }));
    id
}

// ============================================================================
// Geometry helpers
// ============================================================================

fn uv_sphere_mesh(radius: f32, u_segments: u32, v_segments: u32, ctm: Mat4) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for v in 0..=v_segments {
        let phi = std::f32::consts::PI * v as f32 / v_segments as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for u in 0..=u_segments {
            let theta = std::f32::consts::TAU * u as f32 / u_segments as f32;
            let n_local = Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin());
            let pos_local = radius * n_local;
            positions.push(ctm.transform_point3(pos_local));
            normals.push(ctm.transform_vector3(n_local).normalize());
        }
    }

    let row = u_segments + 1;
    for v in 0..v_segments {
        for u in 0..u_segments {
            let i00 = v * row + u;
            let i01 = v * row + u + 1;
            let i10 = (v + 1) * row + u;
            let i11 = (v + 1) * row + u + 1;
            indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }

    TriangleMesh {
        positions,
        indices,
        normals: Some(normals),
        uvs: None,
    }
}

struct QuadCorners {
    p0: Vec3,
    edge_u: Vec3,
    edge_v: Vec3,
}

fn quad_from_corners(p: &[Vec3]) -> QuadCorners {
    QuadCorners {
        p0: p[0],
        edge_u: p[1] - p[0],
        edge_v: p[3] - p[0],
    }
}

fn mat4_from_pbrt_row_major(arr: &[f64]) -> Mat4 {
    // PBRT stores transforms column-major already, matching glm/glam.
    let mut m = [0f32; 16];
    for (i, v) in arr.iter().enumerate() {
        m[i] = *v as f32;
    }
    Mat4::from_cols_array(&m)
}

// ============================================================================
// Parameter accessors
// ============================================================================

fn expect_floats(d: &Directive, n: usize, what: &str) -> Result<Vec<f32>, BuildError> {
    let mut floats = Vec::with_capacity(n);
    for a in &d.positional {
        if let PositionalArg::Float(v) = a {
            floats.push(*v as f32);
        }
    }
    if floats.len() < n {
        return Err(BuildError::Malformed(
            what.into(),
            d.pos.line,
            format!("expected {n} float positionals, got {}", floats.len()),
        ));
    }
    Ok(floats)
}

fn expect_first_string(d: &Directive, what: &str) -> Result<String, BuildError> {
    for a in &d.positional {
        if let PositionalArg::String(s) = a {
            return Ok(s.clone());
        }
    }
    Err(BuildError::Malformed(
        what.into(),
        d.pos.line,
        "expected a positional string (type name)".into(),
    ))
}

fn param_rgb(d: &Directive, name: &str) -> Option<Vec3> {
    let p = d.params.iter().find(|p| p.name == name)?;
    let v = p.as_floats()?;
    if v.len() >= 3 {
        Some(Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32))
    } else if v.len() == 1 {
        Some(Vec3::splat(v[0] as f32))
    } else {
        None
    }
}

fn param_float(d: &Directive, name: &str) -> Option<f32> {
    let p = d.params.iter().find(|p| p.name == name)?;
    first_float(p).map(|v| v as f32)
}

fn first_float(p: &Param) -> Option<f64> {
    match &p.values {
        ParamValues::Floats(v) => v.first().copied(),
        ParamValues::Ints(v) => v.first().map(|i| *i as f64),
        _ => None,
    }
}

fn first_int(p: &Param) -> Option<i64> {
    match &p.values {
        ParamValues::Ints(v) => v.first().copied(),
        ParamValues::Floats(v) => v.first().map(|f| *f as i64),
        _ => None,
    }
}
