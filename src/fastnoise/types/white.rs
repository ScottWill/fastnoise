use glam::{Vec3A, ivec3};

use crate::{Builder, fastnoise::{Sampler, utils::val_coord_3d}};

#[derive(Default, Clone, Copy)]
pub struct WhiteNoiseBuilder {
    frequency: f32,
    seed: u64,
}

impl WhiteNoiseBuilder {
    pub fn frequency(mut self, frequency: f32) -> Self {
        self.frequency = frequency;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Builder for WhiteNoiseBuilder {
    type Output = WhiteNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            frequency: self.frequency,
            seed: self.seed as i32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct WhiteNoise {
    frequency: f32,
    seed: i32,
}

impl Sampler for WhiteNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        self.get_white_noise3d(position.into())
    }
}

impl WhiteNoise {
    fn get_white_noise3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;
        let c = ivec3(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
        );
        val_coord_3d(self.seed, c ^ (c >> 16))
    }
}
