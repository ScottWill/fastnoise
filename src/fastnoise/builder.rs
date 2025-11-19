use super::*;

pub struct FastNoiseBuilder {
    seed: u64,
    frequency: f32,
    interp: Interp,
    noise_type: NoiseType,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    fractal_type: FractalType,
    cellular_distance_function: CellularDistanceFunction,
    cellular_return_type: CellularReturnType,
    cellular_distance_index: (i32, i32),
    cellular_jitter: f32,
    gradient_perturb_amp: f32,
}