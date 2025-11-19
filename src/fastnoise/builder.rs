use super::*;

pub struct FastNoiseBuilder {
    cellular_distance_function: Option<CellularDistanceFunction>,
    cellular_distance_index: Option<(i32, i32)>,
    cellular_jitter: Option<f32>,
    cellular_return_type: Option<CellularReturnType>,
    fractal_type: Option<FractalType>,
    frequency: f32,
    gain: Option<f32>,
    interp: Option<Interp>,
    lacunarity: Option<f32>,
    noise_type: NoiseType,
    octaves: Option<i32>,
    seed: u64,
}

impl FastNoiseBuilder {

}
