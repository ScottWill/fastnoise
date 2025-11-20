use glam::{UVec3, Vec3, vec3a};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::Sampler;
use super::utils::lerp;

pub fn sample_plane<S: Sampler + Sync>(sampler: &S, min: Vec3, max: Vec3, resolution: u32) -> Vec<f32> {
    assert_ne!(0, resolution);
    assert!(min.min(max) == min);
    let side = resolution as usize;
    let side_f32 = side as f32;
    let mut result = Vec::with_capacity(side);
    (0..side)
        .into_par_iter()
        .map(|i| {
            let t = i as f32 / side_f32;
            let v = min.lerp(max, t);
            sampler.sample3d(v)
        })
        .collect_into_vec(&mut result);
    result
}

pub fn sample_cube<S: Sampler + Sync>(sampler: &S, resolution: u32) -> Vec<f32> {
    sample3d(sampler, Vec3::ZERO, Vec3::ONE, UVec3::splat(resolution))
}

pub fn sample3d<S: Sampler + Sync>(sampler: &S, min: Vec3, max: Vec3, resolution: UVec3) -> Vec<f32> {
    assert!(min.min(max) == min);
    assert!(0 < resolution.x);
    assert!(0 < resolution.y);
    assert!(0 < resolution.z);
    let side = resolution.as_usizevec3();
    let side_f32 = {
        let a = resolution.as_vec3();
        vec3a(
            if a.x == 0.0 { 0.0 } else { 1.0 / a.x },
            if a.y == 0.0 { 0.0 } else { 1.0 / a.y },
            if a.z == 0.0 { 0.0 } else { 1.0 / a.z },
        )
    };
    let size = side.element_product();
    let yz = 1.0 / (side_f32.y * side_f32.z);
    let mut result = Vec::with_capacity(size);
    (0..size)
        .into_par_iter()
        .map(|i| {
            let tx = (i % side.y) as f32;
            let ty = (i / side.y % side.z) as f32;
            let tz = i as f32 * yz;
            let t = vec3a(tx, ty, tz) * side_f32;
            let v = vec3a(
                lerp(min.x, max.x, t.x),
                lerp(min.y, max.y, t.y),
                lerp(min.z, max.z, t.z),
            );
            sampler.sample3d(v)
        })
        .collect_into_vec(&mut result);
    result
}