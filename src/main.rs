use fastnoise::*;

fn main() {
    let mut noise = FastNoise::seeded(1337);
    noise.set_noise_type(NoiseType::SimplexFractal);
    noise.set_fractal_type(FractalType::Billow);
    noise.set_interp(Interp::Quintic);
    noise.set_fractal_octaves(5);
    noise.set_fractal_gain(0.6);
    noise.set_fractal_lacunarity(2.0);
    noise.set_frequency(2.0);

    let mut total = 0.0;

    for y in 0..50 {
        for x in 0..80 {
            let n = noise.get_noise((x as f32) / 160.0, (y as f32) / 100.0);
            total += n;
        }
    }

    assert_eq!(-983.51575, total);
    println!("\nTest Passed =)\n");
}