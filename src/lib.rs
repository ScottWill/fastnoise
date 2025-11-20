mod consts;
mod enums;
mod fastnoise;
mod mixer;
mod types;
mod utils;

pub trait Builder {
    type Output;
    fn build(self) -> Self::Output;
}

pub trait Sampler {
    fn sample3d<P>(&self, position: P) -> f32 where P: Into<glam::Vec3A>;
    fn sample<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        let pos = position.into();
        self.sample3d(glam::vec3a(pos.x, pos.y, 0.0))
    }
}

pub use enums::*;
pub use types::cellular::{CellularNoise, CellularNoiseBuilder};
pub use types::cubic::{CubicNoise, CubicNoiseBuilder};
pub use types::perlin::{PerlinNoise, PerlinNoiseBuilder};
pub use types::simplex::{SimplexNoise, SimplexNoiseBuilder};
pub use types::value::{ValueNoise, ValueNoiseBuilder};
pub use types::white::{WhiteNoise, WhiteNoiseBuilder};

#[allow(deprecated)]
pub use fastnoise::FastNoise;

#[cfg(feature="sampler")]
pub mod sampler;

