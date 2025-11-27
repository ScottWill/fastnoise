use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::{Builder, NoiseBuilder, NoiseSampler, Sampler, types::generic::BuilderError};


#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum MixType {
    #[default]
    Avg,
    Max,
    Min,
    SMin(f32),
}

impl Display for MixType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl MixType {
    fn mix(&self, a: f32, b: f32) -> f32 {
        match self {
            MixType::Avg => (a + b) / 2.0,
            MixType::Max => f32::min(a, b),
            MixType::Min => f32::max(a, b),
            MixType::SMin(k) => smin(a, b, *k),
        }
    }
}

// quadratic polynomial
#[inline]
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / (k * 4.0);
    a.min(b) - h * h * k
}

#[derive(Clone, Copy, Default)]
pub struct MixedNoiseBuilder {
    pub mix_type: MixType,
    pub noise0: NoiseBuilder,
    pub noise1: NoiseBuilder,
    pub weights: Option<[f32; 2]>,
}

impl Builder for MixedNoiseBuilder {
    type Output = Result<MixedNoise, BuilderError>;
    fn build(self) -> Self::Output {
        Ok(MixedNoise {
            mix_type: self.mix_type,
            noises: [
                Box::new(self.noise0.try_build_noise_sampler()?),
                Box::new(self.noise1.try_build_noise_sampler()?),
            ],
            weights: match self.weights {
                Some([a, b]) => {
                    if a < 0.0 { return Err(BuilderError::InvalidValue(a.to_string())) }
                    if b < 0.0 { return Err(BuilderError::InvalidValue(b.to_string())) }
                    [a, b]
                },
                None => [1.0; 2],
            },
        })
    }
}

#[derive(Clone)]
pub struct MixedNoise {
    mix_type: MixType,
    noises: [Box<NoiseSampler>; 2],
    weights: [f32; 2],
}

impl Sampler for MixedNoise {
    fn sample3d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec3A> {
        let sample0 = self.noises[0].sample3d(position) * self.weights[0];
        let sample1 = self.noises[1].sample3d(position) * self.weights[1];
        self.mix_type.mix(sample0, sample1)
    }
    fn sample2d<P>(&self, position: P) -> f32 where P: Copy + Into<glam::Vec2> {
        let sample0 = self.noises[0].sample2d(position) * self.weights[0];
        let sample1 = self.noises[1].sample2d(position) * self.weights[1];
        self.mix_type.mix(sample0, sample1)
    }
}