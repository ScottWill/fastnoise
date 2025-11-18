use fastnoise::*;
use glam::{IVec3, Vec3A, vec3a};
use std::time::Instant;

fn main() {
    let mut noise = FastNoise::seeded(1337);
    noise.set_noise_type(NoiseType::Cellular);
    noise.set_cellular_return_type(CellularReturnType::Distance);
    noise.set_fractal_type(FractalType::Billow);
    noise.set_interp(Interp::Quintic);
    noise.set_fractal_octaves(5);
    noise.set_fractal_gain(0.6);
    noise.set_fractal_lacunarity(2.0);
    noise.set_frequency(2.0);

    let mut total = 0.0;
    let then = Instant::now();

    const HS: i32 = 66;
    const VS: Vec3A = Vec3A::splat(HS as f32);

    // let mut i = 0;
    // for y in -HS..HS {
    //     for x in -HS..HS {
    //         for z in -HS..HS {
    //             // let n = noise.get_noise3d(x as f32, y as f32, z as f32); // 1.49s
    //             let n = noise.noise3d(glam::vec3a(x as f32, y as f32, z as f32)); // ?
    //             total += n;
    //             i += 1;
    //         }
    //     }
    // }

    // println!("{i}"); //2299968

    //574992 @230.428125ms
    let total: f32 = sample3d(&noise, -VS, VS, 132).iter().sum();
    // 574992 @50.279833ms

    let now = Instant::now();

    println!("\nTest Passed with {total} @{:?}\n", now - then);

    // a good conversion means this value should remain roughly the same
    // Cellular Natural scalar: 1884.771 @195ms
    // Cellular Natural vector: 1884.771 @222.782041ms !!

    // Cubic scalar: 303.34818 @119ms
    // Cubic vector: 303.34818 @122ms !

    // CubicFractal Billow scalar: -1562558.1 @508ms
    // CubicFractal Billow vector: -1562558.1 @520ms

    // SimplexFractal Billow scalar: -1008301.44 @230ms
    // SimplexFractal Billow vector: -1008301.06 @160ms

    // SimplexFractal FMB scalar: -399.44904 @215ms
    // SimplexFractal FMB vector: -399.45044 @155ms

    // SimplexFractal RigidMulti scalar: -519946.78 @225ms
    // SimplexFractal RigidMulti vector: -519946.78 @155ms

    // Value scalar: 1023.7918 @30ms
    // Value vector: 1023.7918 @38ms !

    // ValueFractal Billow scalar: 10650.566 @100ms
    //                             10650.566 @100ms

    // PerlinFractal RigidMulti scalar: -715432.2 @270ms
    // PerlinFractal RigidMulti vector: -715432.2 @480ms !!

    // WhiteNoise scalar: 354.1004 @10ms
    // WhiteNoise vector: 354.1004 @17.ms !!

}
