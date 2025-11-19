mod builder;
mod consts;
mod enums;
mod fastnoise;
mod sampler;
mod types;
mod utils;

trait Sampler {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A>;
}

pub use enums::*;
pub use fastnoise::FastNoise;
pub use sampler::{sample3d, sample_plane};
