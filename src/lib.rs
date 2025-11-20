mod consts;
mod enums;
mod fastnoise;
mod types;
mod utils;
#[cfg(feature="sampler")]
mod sampler;

pub trait Builder {
    type Output;
    fn build(self) -> Self::Output;
}

pub trait Sampler {
    fn sample3d<P>(&self, position: P) -> f32 where P: Into<glam::Vec3A>;
    fn sample<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        self.sample3d(position)
    }
}

pub use enums::*;
#[allow(deprecated)]
pub use fastnoise::FastNoise;
pub use types::cellular::{CellularNoise, CellularNoiseBuilder};
pub use types::cubic::{CubicNoise, CubicNoiseBuilder};
pub use types::perlin::{PerlinNoise, PerlinNoiseBuilder};
pub use types::simplex::{SimplexNoise, SimplexNoiseBuilder};
pub use types::value::{ValueNoise, ValueNoiseBuilder};
pub use types::white::{WhiteNoise, WhiteNoiseBuilder};
#[cfg(feature="sampler")]
pub use sampler::{sample3d, sample_plane};
