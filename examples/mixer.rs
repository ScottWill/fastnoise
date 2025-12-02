use bracket_color::prelude::*;
use crossterm::queue;
use crossterm::style::{Color::Rgb, Print, SetForegroundColor};
use fastnoise::{Builder as _, BuilderError, FractalNoiseBuilder, FractalType, Interp, MixType, MixedNoiseBuilder, PerlinNoiseBuilder, Sampler as _, SimplexNoiseBuilder, sample2d};
use glam::{Vec2, uvec2};
use std::io::{stdout, Write as _};

fn print_color(color: RGB, text: &str) {
    let foreground = SetForegroundColor(Rgb {
        r: (color.r * 255.0) as u8,
        g: (color.g * 255.0) as u8,
        b: (color.b * 255.0) as u8,
    });
    queue!(stdout(), foreground).expect("set foureground failed");
    queue!(stdout(), Print(text)).expect("print text fail");
}

fn main() -> Result<(), BuilderError> {

    let noise = MixedNoiseBuilder {
        mix_type: MixType::Avg,
        noise0: PerlinNoiseBuilder {
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
            fractal_noise: Some(FractalNoiseBuilder {
                fractal_type: FractalType::RigidMulti,
                gain: 1.3,
                lacunarity: 2.1,
                octaves: 2,
            }),
            frequency: 0.912,
            seed: 8008135,
            ..Default::default()
        }.into(),
        weights: None,
    }.build()?;

    sample2d(&noise, Vec2::ZERO, Vec2::splat(0.5), uvec2(80, 50))
        .iter()
        .enumerate()
        .for_each(|(ix, &n)| {
            if ix > 0 && ix % 80 == 0 {
                print_color(RGB::named(WHITE), "\n");
            }

            if n < 0.0 {
                print_color(RGB::from_f32(0.0, 0.0, 1.0 - (0.0 - n)), "░");
            } else {
                print_color(RGB::from_f32(0.0, n, 0.0), "░");
            }

        });

    print_color(RGB::named(WHITE), "\n");
    stdout().flush().expect("flush failed");

    Ok(())

}
