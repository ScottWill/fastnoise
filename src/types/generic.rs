use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug)]
pub enum BuilderError {
    InvalidNoiseType,
    InvalidValue(String),
    MissingParameter(String),
}

#[derive(Debug, Default, Deserialize, Clone, Copy, Serialize)]
pub struct NoiseBuilder {
    pub cellular_distance_function: Option<CellularDistanceFunction>,
    pub cellular_jitter: Option<f32>,
    pub cellular_return_type: Option<CellularReturnType>,
    pub fractal_type: Option<FractalType>,
    pub frequency: Option<f32>,
    pub gain: Option<f32>,
    pub interp: Option<Interp>,
    pub lacunarity: Option<f32>,
    pub noise_type: Option<NoiseType>,
    pub octaves: Option<u16>,
    pub seed: Option<u64>,
}

impl From<CellularNoiseBuilder> for NoiseBuilder {
    fn from(value: CellularNoiseBuilder) -> Self {
        Self {
            cellular_distance_function: Some(value.cellular_distance_function),
            cellular_jitter: Some(value.cellular_jitter),
            cellular_return_type: Some(value.cellular_return_type),
            frequency: Some(value.frequency),
            noise_type: Some(NoiseType::Cellular),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl From<CubicNoiseBuilder> for NoiseBuilder {
    fn from(value: CubicNoiseBuilder) -> Self {
        Self {
            fractal_type: value.fractal_noise.and_then(|f| Some(f.fractal_type)),
            frequency: Some(value.frequency),
            gain: value.fractal_noise.and_then(|f| Some(f.gain)),
            lacunarity: value.fractal_noise.and_then(|f| Some(f.lacunarity)),
            noise_type: Some(NoiseType::Cubic),
            octaves: value.fractal_noise.and_then(|f| Some(f.octaves)),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl From<PerlinNoiseBuilder> for NoiseBuilder {
    fn from(value: PerlinNoiseBuilder) -> Self {
        Self {
            fractal_type: value.fractal_noise.and_then(|f| Some(f.fractal_type)),
            frequency: Some(value.frequency),
            gain: value.fractal_noise.and_then(|f| Some(f.gain)),
            interp: Some(value.interp),
            lacunarity: value.fractal_noise.and_then(|f| Some(f.lacunarity)),
            noise_type: Some(NoiseType::Perlin),
            octaves: value.fractal_noise.and_then(|f| Some(f.octaves)),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl From<SimplexNoiseBuilder> for NoiseBuilder {
    fn from(value: SimplexNoiseBuilder) -> Self {
        Self {
            fractal_type: value.fractal_noise.and_then(|f| Some(f.fractal_type)),
            frequency: Some(value.frequency),
            gain: value.fractal_noise.and_then(|f| Some(f.gain)),
            lacunarity: value.fractal_noise.and_then(|f| Some(f.lacunarity)),
            noise_type: Some(NoiseType::Simplex),
            octaves: value.fractal_noise.and_then(|f| Some(f.octaves)),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl From<ValueNoiseBuilder> for NoiseBuilder {
    fn from(value: ValueNoiseBuilder) -> Self {
        Self {
            fractal_type: value.fractal_noise.and_then(|f| Some(f.fractal_type)),
            frequency: Some(value.frequency),
            gain: value.fractal_noise.and_then(|f| Some(f.gain)),
            interp: Some(value.interp),
            lacunarity: value.fractal_noise.and_then(|f| Some(f.lacunarity)),
            noise_type: Some(NoiseType::Value),
            octaves: value.fractal_noise.and_then(|f| Some(f.octaves)),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl From<WhiteNoiseBuilder> for NoiseBuilder {
    fn from(value: WhiteNoiseBuilder) -> Self {
        Self {
            frequency: Some(value.frequency),
            noise_type: Some(NoiseType::White),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl NoiseBuilder {
    pub fn try_build_noise_sampler(self) -> Result<NoiseSampler, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cellular) => Ok(NoiseSampler::Cellular(self.try_build_cellular()?)),
            Some(NoiseType::Cubic) => Ok(NoiseSampler::Cubic(self.try_build_cubic()?)),
            Some(NoiseType::Perlin) => Ok(NoiseSampler::Perlin(self.try_build_perlin()?)),
            Some(NoiseType::Simplex) => Ok(NoiseSampler::Simplex(self.try_build_simplex()?)),
            Some(NoiseType::Value) => Ok(NoiseSampler::Value(self.try_build_value()?)),
            Some(NoiseType::White) => Ok(NoiseSampler::White(self.try_build_white()?)),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_cellular(self) -> Result<CellularNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cellular) => Ok(CellularNoiseBuilder {
                cellular_distance_function: self.cellular_distance_function()?,
                cellular_jitter: self.cellular_jitter()?,
                cellular_return_type: self.cellular_return_type()?,
                frequency: self.frequency()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_cubic(self) -> Result<CubicNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cubic) => Ok(CubicNoiseBuilder {
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal)?),
                    None => None,
                },
                frequency: self.frequency()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_perlin(self) -> Result<PerlinNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Perlin) => Ok(PerlinNoiseBuilder {
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal)?),
                    None => None,
                },
                frequency: self.frequency()?,
                interp: self.interp()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_simplex(self) -> Result<SimplexNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Simplex) => Ok(SimplexNoiseBuilder {
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal)?),
                    None => None,
                },
                frequency: self.frequency()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_value(self) -> Result<ValueNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Value) => Ok(ValueNoiseBuilder {
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal)?),
                    None => None,
                },
                frequency: self.frequency()?,
                interp: self.interp()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_white(self) -> Result<WhiteNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::White) => Ok(WhiteNoiseBuilder {
                frequency: self.frequency()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn fractal_noise(&self, fractal_type: FractalType) -> Result<FractalNoiseBuilder, BuilderError> {
        Ok(FractalNoiseBuilder {
            fractal_type,
            gain: self.gain()?,
            lacunarity: self.lacunarity()?,
            octaves: self.octaves()?,
        })
    }

    fn cellular_distance_function(&self) -> Result<CellularDistanceFunction, BuilderError> {
        match self.cellular_distance_function {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_distance_function".to_owned())),
        }
    }
    fn cellular_jitter(&self) -> Result<f32, BuilderError> {
        match self.cellular_jitter {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_jitter".to_owned())),
        }
    }
    fn cellular_return_type(&self) -> Result<CellularReturnType, BuilderError> {
        match self.cellular_return_type {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_return_type".to_owned())),
        }
    }
    fn frequency(&self) -> Result<f32, BuilderError> {
        match self.frequency {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("frequency".to_owned())),
        }
    }
    fn gain(&self) -> Result<f32, BuilderError> {
        match self.gain {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("gain".to_owned())),
        }
    }
    fn interp(&self) -> Result<Interp, BuilderError> {
        match self.interp {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("interp".to_owned())),
        }
    }
    fn lacunarity(&self) -> Result<f32, BuilderError> {
        match self.lacunarity {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("lacunarity".to_owned())),
        }
    }
    fn octaves(&self) -> Result<u16, BuilderError> {
        match self.octaves {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("octaves".to_owned())),
        }
    }
    fn seed(&self) -> Result<u64, BuilderError> {
        match self.seed {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("seed".to_owned())),
        }
    }
}