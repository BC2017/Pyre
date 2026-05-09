//! SAH-binned bounding volume hierarchy.
//!
//! Top-down build: at each node we project primitive centroids onto the longest
//! centroid-bounds axis, bin them, then sweep bin boundaries to find the
//! split with minimum surface-area-heuristic cost. Nodes are emitted in
//! depth-first order so the left child of an interior node is always at
//! `self_index + 1`; the right child's offset is stored explicitly.

use super::SurfaceInteraction;
use crate::math::{Bounds3, Ray};
use glam::Vec3;

const BINS: usize = 12;
const MAX_PRIMS_PER_LEAF: usize = 8;
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECT_COST: f32 = 2.0;

#[derive(Debug, Clone, Copy)]
pub struct BvhNode {
    pub bounds: Bounds3,
    /// Interior: index of the right (second) child node. Leaf: starting index
    /// into the primitive index array.
    pub offset: u32,
    /// 0 → interior. >0 → leaf with that many primitives stored contiguously
    /// at `offset` in the primitive index array.
    pub n_primitives: u16,
    /// Split axis (0/1/2) for interior nodes; used to choose traversal order.
    pub axis: u8,
    _pad: u8,
}

impl BvhNode {
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.n_primitives > 0
    }
}

pub struct Bvh {
    nodes: Vec<BvhNode>,
    primitive_indices: Vec<u32>,
}

impl Bvh {
    pub fn build(prim_bounds: &[Bounds3], prim_centroids: &[Vec3]) -> Self {
        assert_eq!(prim_bounds.len(), prim_centroids.len());
        let n = prim_bounds.len();
        let mut primitive_indices: Vec<u32> = (0..n as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::with_capacity(2 * n.max(1));
        if n > 0 {
            build_recursive(
                &mut nodes,
                &mut primitive_indices,
                prim_bounds,
                prim_centroids,
                0,
                n as u32,
            );
        }
        Bvh {
            nodes,
            primitive_indices,
        }
    }

    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    pub fn primitive_indices(&self) -> &[u32] {
        &self.primitive_indices
    }

    pub fn root_bounds(&self) -> Bounds3 {
        if self.nodes.is_empty() {
            Bounds3::EMPTY
        } else {
            self.nodes[0].bounds
        }
    }

    /// Visit primitive indices that may intersect `ray`, in approximate
    /// near-to-far order. The callback runs Möller–Trumbore (or analogous)
    /// and returns the surface interaction if it hits. The BVH tracks the
    /// closest hit, tightening `t_max` for subsequent visits.
    pub fn intersect<F>(&self, ray: &Ray, mut visit: F) -> Option<SurfaceInteraction>
    where
        F: FnMut(u32, &Ray) -> Option<SurfaceInteraction>,
    {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = Vec3::ONE / ray.direction;
        let dir_neg = [inv_dir.x < 0.0, inv_dir.y < 0.0, inv_dir.z < 0.0];

        let mut closest: Option<SurfaceInteraction> = None;
        let mut t_max = ray.t_max;

        let mut stack: [u32; 64] = [0; 64];
        stack[0] = 0;
        let mut sp: usize = 1;

        while sp > 0 {
            sp -= 1;
            let idx = stack[sp] as usize;
            let node = &self.nodes[idx];

            if !ray_aabb(ray, &node.bounds, inv_dir, t_max) {
                continue;
            }

            if node.is_leaf() {
                let start = node.offset as usize;
                let end = start + node.n_primitives as usize;
                for &prim_idx in &self.primitive_indices[start..end] {
                    let mut tightened = *ray;
                    tightened.t_max = t_max;
                    if let Some(hit) = visit(prim_idx, &tightened) {
                        if hit.t >= ray.t_min && hit.t < t_max {
                            t_max = hit.t;
                            closest = Some(hit);
                        }
                    }
                }
            } else {
                let left = (idx as u32) + 1;
                let right = node.offset;
                // Push the far child first so the near child is visited next.
                if dir_neg[node.axis as usize] {
                    stack[sp] = left;
                    sp += 1;
                    stack[sp] = right;
                    sp += 1;
                } else {
                    stack[sp] = right;
                    sp += 1;
                    stack[sp] = left;
                    sp += 1;
                }
            }
        }
        closest
    }
}

#[inline]
fn ray_aabb(ray: &Ray, b: &Bounds3, inv_dir: Vec3, t_max: f32) -> bool {
    let t1 = (b.min - ray.origin) * inv_dir;
    let t2 = (b.max - ray.origin) * inv_dir;
    let tmin = t1.min(t2).max_element().max(ray.t_min);
    let tmax = t1.max(t2).min_element().min(t_max);
    tmin <= tmax
}

#[inline]
fn axis_value(v: Vec3, a: usize) -> f32 {
    match a {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

fn max_axis(v: Vec3) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

#[derive(Clone, Copy)]
struct Bin {
    count: u32,
    bounds: Bounds3,
}

fn build_recursive(
    nodes: &mut Vec<BvhNode>,
    indices: &mut [u32],
    prim_bounds: &[Bounds3],
    prim_centroids: &[Vec3],
    start: u32,
    end: u32,
) -> u32 {
    let n = (end - start) as usize;

    // Aggregate bounds and centroid bounds for this primitive range.
    let mut bounds = Bounds3::EMPTY;
    let mut centroid_bounds = Bounds3::EMPTY;
    for &i in &indices[start as usize..end as usize] {
        bounds = bounds.union(&prim_bounds[i as usize]);
        centroid_bounds = centroid_bounds.extend(prim_centroids[i as usize]);
    }

    if n == 1 {
        return push_leaf(nodes, bounds, start, 1);
    }

    let cdiag = centroid_bounds.diagonal();
    let axis = max_axis(cdiag);
    let extent = axis_value(cdiag, axis);
    if extent < 1e-9 {
        // All centroids coincident — splitting won't reduce SAH cost.
        return push_leaf(nodes, bounds, start, n.min(u16::MAX as usize) as u16);
    }

    // Bin primitives along the chosen axis.
    let mut bins = [Bin {
        count: 0,
        bounds: Bounds3::EMPTY,
    }; BINS];
    let centroid_min = axis_value(centroid_bounds.min, axis);
    let scale = BINS as f32 / extent;

    for &i in &indices[start as usize..end as usize] {
        let c = axis_value(prim_centroids[i as usize], axis);
        let mut b = ((c - centroid_min) * scale) as i32;
        if b < 0 {
            b = 0;
        }
        if b >= BINS as i32 {
            b = BINS as i32 - 1;
        }
        bins[b as usize].count += 1;
        bins[b as usize].bounds = bins[b as usize].bounds.union(&prim_bounds[i as usize]);
    }

    // Prefix and suffix sweeps for SAH.
    let mut left_count = [0u32; BINS - 1];
    let mut left_bounds = [Bounds3::EMPTY; BINS - 1];
    let mut right_count = [0u32; BINS - 1];
    let mut right_bounds = [Bounds3::EMPTY; BINS - 1];

    let mut acc_count = 0u32;
    let mut acc_bounds = Bounds3::EMPTY;
    for i in 0..BINS - 1 {
        acc_count += bins[i].count;
        acc_bounds = acc_bounds.union(&bins[i].bounds);
        left_count[i] = acc_count;
        left_bounds[i] = acc_bounds;
    }
    let mut acc_count = 0u32;
    let mut acc_bounds = Bounds3::EMPTY;
    for i in (0..BINS - 1).rev() {
        acc_count += bins[i + 1].count;
        acc_bounds = acc_bounds.union(&bins[i + 1].bounds);
        right_count[i] = acc_count;
        right_bounds[i] = acc_bounds;
    }

    let parent_area = bounds.surface_area().max(1e-9);
    let mut best_cost = f32::INFINITY;
    let mut best_split = 0usize;
    for i in 0..BINS - 1 {
        if left_count[i] == 0 || right_count[i] == 0 {
            continue;
        }
        let cost = TRAVERSAL_COST
            + INTERSECT_COST
                * (left_count[i] as f32 * left_bounds[i].surface_area()
                    + right_count[i] as f32 * right_bounds[i].surface_area())
                / parent_area;
        if cost < best_cost {
            best_cost = cost;
            best_split = i;
        }
    }

    let leaf_cost = INTERSECT_COST * n as f32;

    if !best_cost.is_finite() {
        // No bin split was valid (e.g., all primitives in one bin). Force a
        // midpoint partition; if that also fails, fall back to a leaf when
        // small enough, otherwise an equal-count split.
        let mid_pos = centroid_min + 0.5 * extent;
        let mid = partition_by_axis(indices, start, end, prim_centroids, axis, mid_pos);
        if mid != start && mid != end {
            return push_split(
                nodes,
                bounds,
                axis as u8,
                indices,
                prim_bounds,
                prim_centroids,
                start,
                mid,
                end,
            );
        }
        if n <= MAX_PRIMS_PER_LEAF {
            return push_leaf(nodes, bounds, start, n as u16);
        }
        let mid = start + (n as u32) / 2;
        return push_split(
            nodes,
            bounds,
            axis as u8,
            indices,
            prim_bounds,
            prim_centroids,
            start,
            mid,
            end,
        );
    }

    if n <= MAX_PRIMS_PER_LEAF && best_cost >= leaf_cost {
        return push_leaf(nodes, bounds, start, n as u16);
    }

    let split_pos = centroid_min + (best_split + 1) as f32 * extent / BINS as f32;
    let mid = partition_by_axis(indices, start, end, prim_centroids, axis, split_pos);

    if mid == start || mid == end {
        if n <= MAX_PRIMS_PER_LEAF {
            return push_leaf(nodes, bounds, start, n as u16);
        }
        let mid = start + (n as u32) / 2;
        return push_split(
            nodes,
            bounds,
            axis as u8,
            indices,
            prim_bounds,
            prim_centroids,
            start,
            mid,
            end,
        );
    }

    push_split(
        nodes,
        bounds,
        axis as u8,
        indices,
        prim_bounds,
        prim_centroids,
        start,
        mid,
        end,
    )
}

fn push_leaf(nodes: &mut Vec<BvhNode>, bounds: Bounds3, start: u32, n: u16) -> u32 {
    let idx = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds,
        offset: start,
        n_primitives: n,
        axis: 0,
        _pad: 0,
    });
    idx
}

fn push_split(
    nodes: &mut Vec<BvhNode>,
    bounds: Bounds3,
    axis: u8,
    indices: &mut [u32],
    prim_bounds: &[Bounds3],
    prim_centroids: &[Vec3],
    start: u32,
    mid: u32,
    end: u32,
) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        bounds,
        offset: 0,
        n_primitives: 0,
        axis,
        _pad: 0,
    });

    // Left child immediately follows the parent in DFS order.
    let _left = build_recursive(nodes, indices, prim_bounds, prim_centroids, start, mid);
    let right_idx = nodes.len() as u32;
    let _right = build_recursive(nodes, indices, prim_bounds, prim_centroids, mid, end);

    nodes[node_idx as usize].offset = right_idx;
    node_idx
}

fn partition_by_axis(
    indices: &mut [u32],
    start: u32,
    end: u32,
    prim_centroids: &[Vec3],
    axis: usize,
    split_pos: f32,
) -> u32 {
    let mut left = start as usize;
    let mut right = end as usize;
    while left < right {
        let c = axis_value(prim_centroids[indices[left] as usize], axis);
        if c < split_pos {
            left += 1;
        } else {
            right -= 1;
            indices.swap(left, right);
        }
    }
    left as u32
}
