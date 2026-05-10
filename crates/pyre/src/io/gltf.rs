//! glTF 2.0 loader. Walks the default scene's node hierarchy, accumulating
//! transforms, and emits one `TriangleMesh` per primitive with vertex data
//! baked into world space. Quad/strip/fan primitives, textures, and materials
//! are not handled yet — they arrive in later milestones.

use crate::geometry::TriangleMesh;
use glam::{Mat3, Mat4, Vec2, Vec3};
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum GltfError {
    #[error(transparent)]
    Gltf(#[from] ::gltf::Error),
    #[error("primitive has no positions attribute")]
    MissingPositions,
    #[error("non-triangle primitive (mode: {0:?}) — only Triangles is supported")]
    NonTriangleMode(::gltf::mesh::Mode),
}

pub fn load_gltf<P: AsRef<Path>>(path: P) -> Result<Vec<TriangleMesh>, GltfError> {
    let (doc, buffers, _images) = ::gltf::import(path)?;
    let mut out = Vec::new();
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    if let Some(scene) = scene {
        for node in scene.nodes() {
            walk_node(&node, Mat4::IDENTITY, &buffers, &mut out)?;
        }
    }
    Ok(out)
}

fn walk_node(
    node: &::gltf::Node,
    parent: Mat4,
    buffers: &[::gltf::buffer::Data],
    out: &mut Vec<TriangleMesh>,
) -> Result<(), GltfError> {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let xform = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != ::gltf::mesh::Mode::Triangles {
                return Err(GltfError::NonTriangleMode(prim.mode()));
            }
            out.push(read_primitive(&prim, &xform, buffers)?);
        }
    }

    for child in node.children() {
        walk_node(&child, xform, buffers, out)?;
    }
    Ok(())
}

fn read_primitive(
    prim: &::gltf::Primitive,
    xform: &Mat4,
    buffers: &[::gltf::buffer::Data],
) -> Result<TriangleMesh, GltfError> {
    let reader = prim.reader(|b| Some(&buffers[b.index()]));

    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or(GltfError::MissingPositions)?
        .map(|p| xform.transform_point3(Vec3::from(p)))
        .collect();

    let indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    // Normals transform by the inverse-transpose of the upper 3×3 so they stay
    // perpendicular to surfaces under non-uniform scale.
    let normal_xform = Mat3::from_mat4(*xform).inverse().transpose();
    let normals: Option<Vec<Vec3>> = reader.read_normals().map(|iter| {
        iter.map(|n| (normal_xform * Vec3::from(n)).normalize_or_zero())
            .collect()
    });

    let uvs: Option<Vec<Vec2>> = reader
        .read_tex_coords(0)
        .map(|tc| tc.into_f32().map(Vec2::from).collect());

    Ok(TriangleMesh {
        positions,
        indices,
        normals,
        uvs,
    })
}
