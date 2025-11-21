use serde::{Deserialize, Serialize};

use crate::{Builder, CellularDistanceFunction, CellularNoise, CellularNoiseBuilder, CellularReturnType, CubicNoise, CubicNoiseBuilder, FractalType, Interp, NoiseType, PerlinNoise, PerlinNoiseBuilder, SimplexNoise, SimplexNoiseBuilder, ValueNoise, ValueNoiseBuilder, WhiteNoise, WhiteNoiseBuilder};

pub enum BuilderError<'a> {
    InvalidNoiseType,
    MissingParameter(&'a str),
}

#[derive(Debug, Deserialize, Clone, Copy, Serialize)]
pub struct GenericNoiseBuilder {
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

impl GenericNoiseBuilder {
    pub fn try_build_cellular<'a>(self) -> Result<CellularNoise, BuilderError<'a>> {
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

    pub fn try_build_cubic<'a>(self) -> Result<CubicNoise, BuilderError<'a>> {
        match self.noise_type {
            Some(NoiseType::Cubic) => Ok(CubicNoiseBuilder {
                fractal_type: self.fractal_type()?,
                frequency: self.frequency()?,
                gain: self.gain()?,
                lacunarity: self.lacunarity()?,
                octaves: self.octaves()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_perlin<'a>(self) -> Result<PerlinNoise, BuilderError<'a>> {
        match self.noise_type {
            Some(NoiseType::Perlin) => Ok(PerlinNoiseBuilder {
                fractal_type: self.fractal_type()?,
                frequency: self.frequency()?,
                gain: self.gain()?,
                interp: self.interp()?,
                lacunarity: self.lacunarity()?,
                octaves: self.octaves()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_simplex<'a>(self) -> Result<SimplexNoise, BuilderError<'a>> {
        match self.noise_type {
            Some(NoiseType::Simplex) => Ok(SimplexNoiseBuilder {
                fractal_type: self.fractal_type()?,
                frequency: self.frequency()?,
                gain: self.gain()?,
                lacunarity: self.lacunarity()?,
                octaves: self.octaves()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_value<'a>(self) -> Result<ValueNoise, BuilderError<'a>> {
        match self.noise_type {
            Some(NoiseType::Value) => Ok(ValueNoiseBuilder {
                fractal_type: self.fractal_type()?,
                frequency: self.frequency()?,
                gain: self.gain()?,
                interp: self.interp()?,
                lacunarity: self.lacunarity()?,
                octaves: self.octaves()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    pub fn try_build_white<'a>(self) -> Result<WhiteNoise, BuilderError<'a>> {
        match self.noise_type {
            Some(NoiseType::White) => Ok(WhiteNoiseBuilder {
                frequency: self.frequency()?,
                seed: self.seed()?,
            }.build()),
            _ => Err(BuilderError::InvalidNoiseType),
        }
    }

    fn cellular_distance_function<'a>(&self) -> Result<CellularDistanceFunction, BuilderError<'a>> {
        match self.cellular_distance_function {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_distance_function")),
        }
    }
    fn cellular_jitter<'a>(&self) -> Result<f32, BuilderError<'a>> {
        match self.cellular_jitter {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_jitter")),
        }
    }
    fn cellular_return_type<'a>(&self) -> Result<CellularReturnType, BuilderError<'a>> {
        match self.cellular_return_type {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("cellular_return_type")),
        }
    }
    fn fractal_type<'a>(&self) -> Result<FractalType, BuilderError<'a>> {
        match self.fractal_type {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("fractal_type")),
        }
    }
    fn frequency<'a>(&self) -> Result<f32, BuilderError<'a>> {
        match self.frequency {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("frequency")),
        }
    }
    fn gain<'a>(&self) -> Result<f32, BuilderError<'a>> {
        match self.gain {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("gain")),
        }
    }
    fn interp<'a>(&self) -> Result<Interp, BuilderError<'a>> {
        match self.interp {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("interp")),
        }
    }
    fn lacunarity<'a>(&self) -> Result<f32, BuilderError<'a>> {
        match self.lacunarity {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("lacunarity")),
        }
    }
    fn octaves<'a>(&self) -> Result<u16, BuilderError<'a>> {
        match self.octaves {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("octaves")),
        }
    }
    fn seed<'a>(&self) -> Result<u64, BuilderError<'a>> {
        match self.seed {
            Some(value) => Ok(value),
            None => Err(BuilderError::MissingParameter("seed")),
        }
    }
}