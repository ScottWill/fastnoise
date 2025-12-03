use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::{Builder, NoiseBuilder, NoiseSampler, Sampler, types::generic::BuilderError};


#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum MixType {
    #[default]
    Add,
    Average,
    Maximum,
    Minimum,
    Subtract,
}

impl Display for MixType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl MixType {
    fn mix(&self, a: f32, b: f32) -> f32 {
        match self {
            MixType::Add => a + b,
            MixType::Average => (a + b) * 0.5,
            MixType::Maximum => f32::max(a, b),
            MixType::Minimum => f32::min(a, b),
            MixType::Subtract => a - b,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MixedNoiseBuilder {
    pub mix_type: MixType,
    pub noise0: NoiseBuilder,
    pub noise1: NoiseBuilder,
}

impl Builder for MixedNoiseBuilder {
    type Output = Result<MixedNoise, BuilderError>;
    fn build(self) -> Self::Output {
        Ok(MixedNoise {
            mix_type: self.mix_type,
            noises: Box::new([
                self.noise0.try_build_noise_sampler()?,
                self.noise1.try_build_noise_sampler()?,
            ]),
        })
    }
}

#[derive(Clone, Debug)]
pub struct MixedNoise {
    mix_type: MixType,
    noises: Box<[NoiseSampler; 2]>,
}

impl Sampler for MixedNoise {
    fn sample3d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec3A> {
        let sample0 = self.noises[0].sample3d(position);
        let sample1 = self.noises[1].sample3d(position);
        self.mix_type.mix(sample0, sample1)
    }
    fn sample2d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec2> {
        let sample0 = self.noises[0].sample2d(position);
        let sample1 = self.noises[1].sample2d(position);
        self.mix_type.mix(sample0, sample1)
    }
}