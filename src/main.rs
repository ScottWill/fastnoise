use std::time::Instant;

use fastnoise::{FastNoise, FractalType};


fn main() {

    let mut noise = FastNoise::default();
    noise.set_fractal_type(FractalType::FBM);

    let then = Instant::now();
    let mut samples = vec![];

    for x in 0..1000 {
        for y in 0..1000 {
            for z in 0..100 {
                samples.push(noise.get_noise3d([x as f32, y as f32, z as f32]));
            }
        }
    }

    let now = Instant::now();

    println!("{:?}", now - then);

}