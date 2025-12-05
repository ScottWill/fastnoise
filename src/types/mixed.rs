use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

use crate::{Builder, NoiseSampler, Sampler, types::generic::BuilderError};

#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MixedNoiseBuilder {
    pub amplitude: f32,
    pub mix_type: MixType,
    pub noise0: NoiseSampler,
    pub noise1: NoiseSampler,
}

impl Default for MixedNoiseBuilder {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            mix_type: Default::default(),
            noise0: Default::default(),
            noise1: Default::default(),
        }
    }
}

impl Builder for MixedNoiseBuilder {
    type Output = Result<MixedNoise, BuilderError>;
    fn build(self) -> Self::Output {
        Ok(MixedNoise {
            amplitude: self.amplitude,
            mix_type: self.mix_type,
            noises: [
                Box::new(self.noise0),
                Box::new(self.noise1),
            ],
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MixedNoise {
    amplitude: f32,
    mix_type: MixType,
    noises: [Box<NoiseSampler>; 2],
}

impl Sampler for MixedNoise {
    fn sample3d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec3A> {
        let sample0 = self.noises[0].sample3d(position);
        let sample1 = self.noises[1].sample3d(position);
        self.mix_type.mix(sample0, sample1) * self.amplitude
    }
    fn sample2d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec2> {
        let sample0 = self.noises[0].sample2d(position);
        let sample1 = self.noises[1].sample2d(position);
        self.mix_type.mix(sample0, sample1) * self.amplitude
    }
}
