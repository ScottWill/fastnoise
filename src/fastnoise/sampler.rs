use glam::{Vec3, vec3a};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{FastNoise, utils::lerp};

pub fn sample_plane(noise: &FastNoise, min: Vec3, max: Vec3, resolution: u32) -> Vec<f32> {
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
            noise.noise3d(v)
        })
        .collect_into_vec(&mut result);
    result
}

pub fn sample3d(noise: &FastNoise, min: Vec3, max: Vec3, resolution: u32) -> Vec<f32> {
    assert_ne!(0, resolution);
    assert!(min.min(max) == min);
    let side = resolution as usize;
    let side_f32 = side as f32;
    let size = side.pow(3);
    let mut result = Vec::with_capacity(size);
    (0..size)
        .into_par_iter()
        .map(|i| {
            let tx = (i % side) as f32 / side_f32;
            let ty = (i / side % side) as f32 / side_f32;
            let tz = (i / (side * side)) as f32 / side_f32;
            let v = vec3a(
                lerp(min.x, max.x, tx),
                lerp(min.y, max.y, ty),
                lerp(min.z, max.z, tz),
            );
            noise.noise3d(v)
        })
        .collect_into_vec(&mut result);
    result
}