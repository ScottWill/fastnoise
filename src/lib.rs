mod consts;
mod enums;
mod fastnoise;
mod mixer;
mod sampler;
mod types;
mod utils;

pub trait Builder {
    type Output;
    fn build(self) -> Self::Output;
}

pub trait Sampler {
    fn sample3d<P>(&self, position: P) -> f32 where P: Into<glam::Vec3A>;
    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        self.sample3d(position.into().extend(0.0))
    }
}

pub use enums::{CellularDistanceFunction, CellularReturnType, FractalType, Interp, NoiseType};
pub use sampler::{sample_cube, sample_plane, sample3d};
pub use types::cellular::{CellularNoise, CellularNoiseBuilder};
pub use types::cubic::{CubicNoise, CubicNoiseBuilder};
pub use types::perlin::{PerlinNoise, PerlinNoiseBuilder};
pub use types::simplex::{SimplexNoise, SimplexNoiseBuilder};
pub use types::value::{ValueNoise, ValueNoiseBuilder};
pub use types::white::{WhiteNoise, WhiteNoiseBuilder};

#[allow(deprecated)]
pub use fastnoise::FastNoise;
