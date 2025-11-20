use crate::{Builder, CellularDistanceFunction, CellularNoise, CellularNoiseBuilder, CellularReturnType, CubicNoise, CubicNoiseBuilder, FractalType, Interp, NoiseType, PerlinNoise, PerlinNoiseBuilder, SimplexNoise, SimplexNoiseBuilder, ValueNoise, ValueNoiseBuilder, WhiteNoise, WhiteNoiseBuilder};

pub struct InvalidNoiseType;

pub struct GenericNoiseBuilder {
    pub cellular_distance_function: CellularDistanceFunction,
    pub cellular_jitter: f32,
    pub cellular_return_type: CellularReturnType,
    pub fractal_type: Option<FractalType>,
    pub frequency: f32,
    pub gain: f32,
    pub interp: Interp,
    pub lacunarity: f32,
    pub noise_type: NoiseType,
    pub octaves: u16,
    pub seed: u64,
}

impl GenericNoiseBuilder {
    pub fn try_build_cellular(self) -> Result<CellularNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::Cellular => Ok(CellularNoiseBuilder {
                cellular_distance_function: self.cellular_distance_function,
                cellular_jitter: self.cellular_jitter,
                cellular_return_type: self.cellular_return_type,
                frequency: self.frequency,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }

    pub fn try_build_cubic(self) -> Result<CubicNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::Cubic => Ok(CubicNoiseBuilder {
                fractal_type: self.fractal_type,
                frequency: self.frequency,
                gain: self.gain,
                lacunarity: self.lacunarity,
                octaves: self.octaves,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }

    pub fn try_build_perlin(self) -> Result<PerlinNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::Perlin => Ok(PerlinNoiseBuilder {
                fractal_type: self.fractal_type,
                frequency: self.frequency,
                gain: self.gain,
                interp: self.interp,
                lacunarity: self.lacunarity,
                octaves: self.octaves,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }

    pub fn try_build_simplex(self) -> Result<SimplexNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::Simplex => Ok(SimplexNoiseBuilder {
                fractal_type: self.fractal_type,
                frequency: self.frequency,
                gain: self.gain,
                lacunarity: self.lacunarity,
                octaves: self.octaves,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }

    pub fn try_build_value(self) -> Result<ValueNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::Value => Ok(ValueNoiseBuilder {
                fractal_type: self.fractal_type,
                frequency: self.frequency,
                gain: self.gain,
                interp: self.interp,
                lacunarity: self.lacunarity,
                octaves: self.octaves,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }

    pub fn try_build_white(self) -> Result<WhiteNoise, InvalidNoiseType> {
        match self.noise_type {
            NoiseType::White => Ok(WhiteNoiseBuilder {
                frequency: self.frequency,
                seed: self.seed,
            }.build()),
            _ => Err(InvalidNoiseType),
        }
    }
}