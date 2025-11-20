use bracket_color::prelude::*;
use crossterm::queue;
use crossterm::style::{Color::Rgb, Print, SetForegroundColor};
use fastnoise::{Builder as _, FractalType, PerlinNoiseBuilder, Sampler as _};
use std::io::{stdout, Write as _};

fn print_color(color: RGB, text: &str) {
    queue!(
        stdout(),
        SetForegroundColor(Rgb {
            r: (color.r * 255.0) as u8,
            g: (color.g * 255.0) as u8,
            b: (color.b * 255.0) as u8,
        })
    )
    .expect("Command Fail");
    queue!(stdout(), Print(text)).expect("Command fail");
}

fn main() {
    let noise = PerlinNoiseBuilder {
        fractal_type: Some(FractalType::FBM),
        frequency: 2.0,
        gain: 0.6,
        lacunarity: 2.0,
        octaves: 5,
        seed: 31337,
        ..Default::default()
    }.build();

    for y in 0..50 {
        for x in 0..80 {
            let n = noise.sample2d([(x as f32) / 160.0, (y as f32) / 100.0]);
            if n < 0.0 {
                print_color(RGB::from_f32(0.0, 0.0, 1.0 - (0.0 - n)), "░");
            } else {
                print_color(RGB::from_f32(0.0, n, 0.0), "░");
            }
        }
        print_color(RGB::named(WHITE), "\n");
    }

    print_color(RGB::named(WHITE), "\n");
    stdout().flush().expect("Flush Fail");
}
