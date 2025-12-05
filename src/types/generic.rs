use serde::{Deserialize, Serialize};

use crate::*;

#[derive(Debug)]
pub enum BuilderError {
    InvalidNoiseType,
    InvalidValue(String),
    MissingParameter(String),
}

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
pub struct NoiseBuilder {
    pub amplitude: Option<f32>,
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
            amplitude: Some(value.amplitude),
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
            amplitude: Some(value.amplitude),
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
            amplitude: Some(value.amplitude),
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
            amplitude: Some(value.amplitude),
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
            amplitude: Some(value.amplitude),
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
            amplitude: Some(value.amplitude),
            frequency: Some(value.frequency),
            noise_type: Some(NoiseType::White),
            seed: Some(value.seed),
            ..Default::default()
        }
    }
}

impl NoiseBuilder {
    pub fn build_noise_sampler(self) -> NoiseSampler {
        self.into_noise_sampler(true).unwrap()
    }

    pub fn build_cellular(self) -> CellularNoise {
        self.into_cellular(true).unwrap()
    }

    pub fn build_cubic(self) -> CubicNoise {
        self.into_cubic(true).unwrap()
    }

    pub fn build_perlin(self) -> PerlinNoise {
        self.into_perlin(true).unwrap()
    }

    pub fn build_simplex(self) -> SimplexNoise {
        self.into_simplex(true).unwrap()
    }

    pub fn build_value(self) -> ValueNoise {
        self.into_value(true).unwrap()
    }

    pub fn build_white(self) -> WhiteNoise {
        self.into_white(true).unwrap()
    }

    pub fn try_build_noise_sampler(self) -> Result<NoiseSampler, BuilderError> {
        self.into_noise_sampler(false)
    }

    pub fn try_build_cellular(self) -> Result<CellularNoise, BuilderError> {
        self.into_cellular(false)
    }

    pub fn try_build_cubic(self) -> Result<CubicNoise, BuilderError> {
        self.into_cubic(false)
    }

    pub fn try_build_perlin(self) -> Result<PerlinNoise, BuilderError> {
        self.into_perlin(false)
    }

    pub fn try_build_simplex(self) -> Result<SimplexNoise, BuilderError> {
        self.into_simplex(false)
    }

    pub fn try_build_value(self) -> Result<ValueNoise, BuilderError> {
        self.into_value(false)
    }

    pub fn try_build_white(self) -> Result<WhiteNoise, BuilderError> {
        self.into_white(false)
    }

    fn into_noise_sampler(self, use_default: bool) -> Result<NoiseSampler, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cellular) => Ok(NoiseSampler::Cellular(self.into_cellular(use_default)?)),
            Some(NoiseType::Cubic) => Ok(NoiseSampler::Cubic(self.into_cubic(use_default)?)),
            Some(NoiseType::Perlin) => Ok(NoiseSampler::Perlin(self.into_perlin(use_default)?)),
            Some(NoiseType::Simplex) => Ok(NoiseSampler::Simplex(self.into_simplex(use_default)?)),
            Some(NoiseType::Value) => Ok(NoiseSampler::Value(self.into_value(use_default)?)),
            Some(NoiseType::White) => Ok(NoiseSampler::White(self.into_white(use_default)?)),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_cellular(self, use_default: bool) -> Result<CellularNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cellular) => Ok(CellularNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                cellular_distance_function: self.cellular_distance_function(use_default)?,
                cellular_jitter: self.cellular_jitter(use_default)?,
                cellular_return_type: self.cellular_return_type(use_default)?,
                frequency: self.frequency(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_cubic(self, use_default: bool) -> Result<CubicNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Cubic) => Ok(CubicNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal, use_default)?),
                    None => None,
                },
                frequency: self.frequency(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_perlin(self, use_default: bool) -> Result<PerlinNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Perlin) => Ok(PerlinNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal, use_default)?),
                    None => None,
                },
                frequency: self.frequency(use_default)?,
                interp: self.interp(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_simplex(self, use_default: bool) -> Result<SimplexNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Simplex) => Ok(SimplexNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal, use_default)?),
                    None => None,
                },
                frequency: self.frequency(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_value(self, use_default: bool) -> Result<ValueNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::Value) => Ok(ValueNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                fractal_noise: match self.fractal_type {
                    Some(fractal) => Some(self.fractal_noise(fractal, use_default)?),
                    None => None,
                },
                frequency: self.frequency(use_default)?,
                interp: self.interp(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn into_white(self, use_default: bool) -> Result<WhiteNoise, BuilderError> {
        match self.noise_type {
            Some(NoiseType::White) => Ok(WhiteNoiseBuilder {
                amplitude: self.amplitude(use_default)?,
                frequency: self.frequency(use_default)?,
                seed: self.seed(use_default)?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn amplitude(&self, use_default: bool) -> Result<f32, BuilderError> {
        match self.amplitude {
            Some(value) => Ok(value),
            None if use_default => Ok(1.0),
            _ => Err(BuilderError::MissingParameter("amplitude".to_owned())),
        }
    }

    fn fractal_noise(&self, fractal_type: FractalType, use_default: bool) -> Result<FractalNoiseBuilder, BuilderError> {
        Ok(FractalNoiseBuilder {
            fractal_type,
            gain: self.gain(use_default)?,
            lacunarity: self.lacunarity(use_default)?,
            octaves: self.octaves(use_default)?,
        })
    }

    fn cellular_distance_function(&self, use_default: bool) -> Result<CellularDistanceFunction, BuilderError> {
        match self.cellular_distance_function {
            Some(value) => Ok(value),
            None if use_default => Ok(Default::default()),
            _ => Err(BuilderError::MissingParameter("cellular_distance_function".to_owned())),
        }
    }
    fn cellular_jitter(&self, use_default: bool) -> Result<f32, BuilderError> {
        match self.cellular_jitter {
            Some(value) => Ok(value),
            None if use_default => Ok(Default::default()),
            _ => Err(BuilderError::MissingParameter("cellular_jitter".to_owned())),
        }
    }
    fn cellular_return_type(&self, use_default: bool) -> Result<CellularReturnType, BuilderError> {
        match self.cellular_return_type {
            Some(value) => Ok(value),
            None if use_default => Ok(Default::default()),
            _ => Err(BuilderError::MissingParameter("cellular_return_type".to_owned())),
        }
    }
    fn frequency(&self, use_default: bool) -> Result<f32, BuilderError> {
        match self.frequency {
            Some(value) => Ok(value),
            None if use_default => Ok(1.0),
            _ => Err(BuilderError::MissingParameter("frequency".to_owned())),
        }
    }
    fn gain(&self, use_default: bool) -> Result<f32, BuilderError> {
        match self.gain {
            Some(value) => Ok(value),
            None if use_default => Ok(1.0),
            _ => Err(BuilderError::MissingParameter("gain".to_owned())),
        }
    }
    fn interp(&self, use_default: bool) -> Result<Interp, BuilderError> {
        match self.interp {
            Some(value) => Ok(value),
            None if use_default => Ok(Default::default()),
            _ => Err(BuilderError::MissingParameter("interp".to_owned())),
        }
    }
    fn lacunarity(&self, use_default: bool) -> Result<f32, BuilderError> {
        match self.lacunarity {
            Some(value) => Ok(value),
            None if use_default => Ok(1.0),
            _ => Err(BuilderError::MissingParameter("lacunarity".to_owned())),
        }
    }
    fn octaves(&self, use_default: bool) -> Result<u16, BuilderError> {
        match self.octaves {
            Some(value) => Ok(value),
            None if use_default => Ok(1),
            _ => Err(BuilderError::MissingParameter("octaves".to_owned())),
        }
    }
    fn seed(&self, use_default: bool) -> Result<u64, BuilderError> {
        match self.seed {
            Some(value) => Ok(value),
            None if use_default => Ok(0),
            _ => Err(BuilderError::MissingParameter("seed".to_owned())),
        }
    }
}