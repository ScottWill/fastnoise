use glam::{Vec2, Vec3A, ivec2, ivec3};
#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};

use crate::{Builder, Sampler, utils::{val_coord_2d, val_coord_3d}};

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct WhiteNoiseBuilder {
    pub amplitude: f32,
    pub frequency: f32,
    pub seed: u64,
}

impl Default for WhiteNoiseBuilder {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            frequency: 1.0,
            seed: Default::default(),
        }
    }
}

impl Builder for WhiteNoiseBuilder {
    type Output = WhiteNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            amplitude: self.amplitude,
            frequency: self.frequency,
            seed: self.seed as i32,
        }
    }
}

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct WhiteNoise {
    amplitude: f32,
    frequency: f32,
    seed: i32,
}

impl From<WhiteNoiseBuilder> for WhiteNoise {
    fn from(value: WhiteNoiseBuilder) -> Self {
        value.build()
    }
}

impl Sampler for WhiteNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        self.get_white_noise3d(position.into()) * self.amplitude
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        self.get_white_noise(position.into()) * self.amplitude
    }

}

impl WhiteNoise {
    #[inline]
    fn get_white_noise3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;
        let c = ivec3(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
        );
        val_coord_3d(self.seed, c ^ c >> 16)
    }

    #[inline]
    fn get_white_noise(&self, pos: Vec2) -> f32 {
        let coord = ivec2(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
        );
        val_coord_2d(self.seed, coord ^ coord >> 16)
    }
}
