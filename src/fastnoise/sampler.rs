use glam::{Vec3A, vec3a};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use super::{FastNoise, utils::lerp};

pub fn sample3d(noise: &FastNoise, min: Vec3A, max: Vec3A, resolution: u32) -> Vec<f32> {
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