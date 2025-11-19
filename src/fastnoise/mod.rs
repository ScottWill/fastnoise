mod consts;
mod enums;
mod fastnoise;
mod sampler;
mod types;
mod utils;

pub trait Builder {
    type Output;
    fn build(self) -> Self::Output;
}

pub trait Sampler {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A>;
}

pub use enums::*;
pub use fastnoise::FastNoise;
pub use sampler::{sample3d, sample_plane};
pub use types::cellular::{CellularNoise, CellularNoiseBuilder};
pub use types::cubic::{CubicNoise, CubicNoiseBuilder};
pub use types::perlin::{PerlinNoise, PerlinNoiseBuilder};
pub use types::simplex::{SimplexNoise, SimplexNoiseBuilder};
pub use types::value::{ValueNoise, ValueNoiseBuilder};
pub use types::white::{WhiteNoise, WhiteNoiseBuilder};
