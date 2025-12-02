use glam::{Vec2, Vec3A, ivec2, ivec3};

use crate::{Builder, Sampler, traits::Domain, utils::{val_coord_2d, val_coord_3d}};

#[derive(Clone, Copy, Debug, Default)]
pub struct WhiteNoiseBuilder {
    pub domain: Option<[f32; 2]>,
    pub frequency: f32,
    pub seed: u64,
}

impl Builder for WhiteNoiseBuilder {
    type Output = WhiteNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            domain: self.domain.and_then(|[a, b]| Some([a, b - a])),
            frequency: self.frequency,
            seed: self.seed as i32,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WhiteNoise {
    domain: Option<[f32; 2]>,
    frequency: f32,
    seed: i32,
}

impl From<WhiteNoiseBuilder> for WhiteNoise {
    fn from(value: WhiteNoiseBuilder) -> Self {
        value.build()
    }
}

impl Domain for WhiteNoise {
    fn in_domain(&self, value: f32) -> f32 {
        match self.domain {
            Some([a, b]) => a + b * (value + 1.0) * 0.5,
            None => value,
        }
    }
}

impl Sampler for WhiteNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let value = self.get_white_noise3d(position.into());
        self.in_domain(value)
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        let value = self.get_white_noise(position.into());
        self.in_domain(value)
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
