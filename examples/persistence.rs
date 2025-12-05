use fastnoise::*;
use std::f32::consts::PI;

fn main() -> Result<(), BuilderError> {

    let noise = NoiseSampler::Mixed(MixedNoiseBuilder {
        amplitude: PI * 0.5,
        mix_type: MixType::Subtract,
        noise0: PerlinNoiseBuilder {
            amplitude: 3.0,
            fractal_noise: Some(FractalNoiseBuilder {
                fractal_type: FractalType::FBM,
                gain: 0.6,
                lacunarity: 1.2,
                octaves: 4,
            }),
            frequency: 1.12,
            interp: Interp::Quintic,
            seed: 31337,
            ..Default::default()
        }.into(),
        noise1: SimplexNoiseBuilder {
            amplitude: 0.8,
            fractal_noise: Some(FractalNoiseBuilder {
                fractal_type: FractalType::RigidMulti,
                gain: 1.3,
                lacunarity: 2.1,
                octaves: 2,
            }),
            frequency: 0.912,
            seed: 8008135,
        }.into(),
    }.build()?);

    let r = ron::to_string(&noise).expect("should have worked");

    println!("{r}");

    let s: NoiseSampler = ron::from_str(&r).expect("should have worked");

    println!("{s:?}");

    Ok(())
}