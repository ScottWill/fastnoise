use fastnoise::*;
use std::time::Instant;

fn main() {
    let mut noise = FastNoise::seeded(1337);
    noise.set_noise_type(NoiseType::SimplexFractal);
    noise.set_fractal_type(FractalType::RigidMulti);
    noise.set_interp(Interp::Quintic);
    noise.set_fractal_octaves(5);
    noise.set_fractal_gain(0.6);
    noise.set_fractal_lacunarity(2.0);
    noise.set_frequency(2.0);

    let mut total = 0.0;
    let then = Instant::now();

    const HS: i32 = 66;

    for y in -HS..HS {
        for x in -HS..HS {
            for z in -HS..HS {
                // let n = noise.get_noise3d(x as f32, y as f32, z as f32); // 1.49s
                let n = noise.get_noise3d_vec(glam::vec3a(x as f32, y as f32, z as f32)); // ?
                total += n;
            }
        }
    }

    let now = Instant::now();

    // a good conversion means this value should remain roughly the same
    // SimplexFractal Billow scalar: -1008301.44 @230ms
    // SimplexFractal Billow vector: -1008301.06 @160ms

    // SimplexFractal FMB scalar: -399.44904 @215ms
    // SimplexFractal FMB vector: -399.45044 @155ms

    // SimplexFractal RigidMulti scalar: -519946.78 @225ms
    // SimplexFractal RigidMulti vector: -519946.78 @155ms
    println!("\nTest Passed with {total} =) in {:?}\n", now - then);

}
