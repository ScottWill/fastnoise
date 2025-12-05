use glam::{UVec2, UVec3, Vec2, Vec3};
#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};

use crate::{types::mixed::MixedNoise, *};
use super::utils::lerp;

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub enum NoiseSampler {
    Cellular(CellularNoise),
    Cubic(CubicNoise),
    Mixed(MixedNoise),
    Perlin(PerlinNoise),
    Simplex(SimplexNoise),
    Value(ValueNoise),
    White(WhiteNoise),
    #[default]
    None,
}

impl Sampler for NoiseSampler {
    fn sample3d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec3A> {
        match self {
            NoiseSampler::Cellular(noise) => noise.sample3d(position),
            NoiseSampler::Cubic(noise) => noise.sample3d(position),
            NoiseSampler::Mixed(noise) => noise.sample3d(position),
            NoiseSampler::Perlin(noise) => noise.sample3d(position),
            NoiseSampler::Simplex(noise) => noise.sample3d(position),
            NoiseSampler::Value(noise) => noise.sample3d(position),
            NoiseSampler::White(noise) => noise.sample3d(position),
            NoiseSampler::None => 0.0,
        }
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec2> {
        match self {
            NoiseSampler::Cellular(noise) => noise.sample2d(position),
            NoiseSampler::Cubic(noise) => noise.sample2d(position),
            NoiseSampler::Mixed(noise) => noise.sample2d(position),
            NoiseSampler::Perlin(noise) => noise.sample2d(position),
            NoiseSampler::Simplex(noise) => noise.sample2d(position),
            NoiseSampler::Value(noise) => noise.sample2d(position),
            NoiseSampler::White(noise) => noise.sample2d(position),
            NoiseSampler::None => 0.0,
        }
    }
}

impl From<CellularNoiseBuilder> for NoiseSampler {
    fn from(value: CellularNoiseBuilder) -> Self {
        Self::Cellular(value.build())
    }
}

impl From<CellularNoise> for NoiseSampler {
    fn from(value: CellularNoise) -> Self {
        Self::Cellular(value)
    }
}

impl From<CubicNoiseBuilder> for NoiseSampler {
    fn from(value: CubicNoiseBuilder) -> Self {
        Self::Cubic(value.build())
    }
}

impl From<CubicNoise> for NoiseSampler {
    fn from(value: CubicNoise) -> Self {
        Self::Cubic(value)
    }
}

impl From<PerlinNoiseBuilder> for NoiseSampler {
    fn from(value: PerlinNoiseBuilder) -> Self {
        Self::Perlin(value.build())
    }
}

impl From<PerlinNoise> for NoiseSampler {
    fn from(value: PerlinNoise) -> Self {
        Self::Perlin(value)
    }
}

impl From<SimplexNoiseBuilder> for NoiseSampler {
    fn from(value: SimplexNoiseBuilder) -> Self {
        Self::Simplex(value.build())
    }
}

impl From<SimplexNoise> for NoiseSampler {
    fn from(value: SimplexNoise) -> Self {
        Self::Simplex(value)
    }
}

impl From<ValueNoiseBuilder> for NoiseSampler {
    fn from(value: ValueNoiseBuilder) -> Self {
        Self::Value(value.build())
    }
}

impl From<ValueNoise> for NoiseSampler {
    fn from(value: ValueNoise) -> Self {
        Self::Value(value)
    }
}

impl From<WhiteNoiseBuilder> for NoiseSampler {
    fn from(value: WhiteNoiseBuilder) -> Self {
        Self::White(value.build())
    }
}

impl From<WhiteNoise> for NoiseSampler {
    fn from(value: WhiteNoise) -> Self {
        Self::White(value)
    }
}

#[cfg(feature = "parallel")]
pub fn sample2d<S: Sampler + Sync>(sampler: &S, min: Vec2, max: Vec2, resolution: UVec2) -> Vec<f32> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let resolution = resolution.as_usizevec2();
    let size = resolution.element_product();
    let mut result = Vec::with_capacity(size);
    (0..size)
        .into_par_iter()
        .map(|ix| {
            let tx = (ix % resolution.x) as f32 / resolution.x as f32;
            let ty = (ix / resolution.x) as f32 / resolution.y as f32;
            let pos = Vec2::new(
                lerp(min.x, max.x, tx),
                lerp(min.y, max.y, ty),
            );
            let sample = sampler.sample2d(pos);
            sample
        })
        .collect_into_vec(&mut result);

    result

}

#[cfg(not(feature = "parallel"))]
pub fn sample2d<S: Sampler + Sync>(sampler: &S, min: Vec2, max: Vec2, resolution: UVec2) -> Vec<f32> {
    let resolution = resolution.as_usizevec2();
    let size = resolution.element_product();
    (0..size)
        .into_iter()
        .map(|ix| {
            let tx = (ix % resolution.x) as f32 / resolution.x as f32;
            let ty = (ix / resolution.x) as f32 / resolution.y as f32;
            let pos = Vec2::new(
                lerp(min.x, max.x, tx),
                lerp(min.y, max.y, ty),
            );
            let sample = sampler.sample2d(pos);
            sample
        })
        .collect()
}

pub fn sample_plane<S: Sampler + Sync>(sampler: &S, resolution: u32) -> Vec<f32> {
    sample2d(sampler, Vec2::ZERO, Vec2::ONE, UVec2::splat(resolution))
}

pub fn sample_cube<S: Sampler + Sync>(sampler: &S, resolution: u32) -> Vec<f32> {
    sample3d(sampler, Vec3::ZERO, Vec3::ONE, UVec3::splat(resolution))
}

#[cfg(feature = "parallel")]
pub fn sample3d<S: Sampler + Sync>(sampler: &S, min: Vec3, max: Vec3, resolution: UVec3) -> Vec<f32> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    let resolution = resolution.as_usizevec3();
    let size = resolution.element_product();
    let mut result = Vec::with_capacity(size);
    (0..size)
        .into_par_iter()
        .map(|ix| {
            let tx = (ix % resolution.x) as f32 / resolution.x as f32;
            let ty = (ix / (resolution.x % resolution.z)) as f32 / resolution.y as f32;
            let tz = (ix / (resolution.x * resolution.z)) as f32 / resolution.z as f32;
            let pos = Vec3::new(
                lerp(min.x, max.x, tx),
                lerp(min.y, max.y, ty),
                lerp(min.z, max.z, tz),
            );
            let sample = sampler.sample3d(pos);
            sample
        })
        .collect_into_vec(&mut result);

    result
}

#[cfg(not(feature = "parallel"))]
pub fn sample3d<S: Sampler + Sync>(sampler: &S, min: Vec3, max: Vec3, resolution: UVec3) -> Vec<f32> {
    let resolution = resolution.as_usizevec3();
    let size = resolution.element_product();
    (0..size)
        .into_iter()
        .map(|ix| {
            let tx = (ix % resolution.x) as f32 / resolution.x as f32;
            let ty = (ix / (resolution.x % resolution.z)) as f32 / resolution.y as f32;
            let tz = (ix / (resolution.x * resolution.z)) as f32 / resolution.z as f32;
            let pos = Vec3::new(
                lerp(min.x, max.x, tx),
                lerp(min.y, max.y, ty),
                lerp(min.z, max.z, tz),
            );
            let sample = sampler.sample3d(pos);
            sample
        })
        .collect()
}