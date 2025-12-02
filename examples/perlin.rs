use bracket_color::prelude::*;
use crossterm::queue;
use crossterm::style::{Color::Rgb, Print, SetForegroundColor};
use fastnoise::{Builder as _, FractalNoiseBuilder, FractalType, PerlinNoiseBuilder, Sampler as _};
use std::f32::consts::PI;
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

fn main() {
    let noise = PerlinNoiseBuilder {
        frequency: PI,
        seed: 8008135,
        ..Default::default()
    }.build();

    let mut max = f32::MIN;
    let mut min = f32::MAX;

    for y in 0..50 {
        for x in 0..80 {
            let n = noise.sample2d([(x as f32) / 160.0, (y as f32) / 100.0]) - 0.5;

            max = max.max(n);
            min = min.min(n);

            if n < 0.0 {
                print_color(RGB::from_f32(0.0, 0.0, 1.0 - (0.0 - n)), "░");
            } else {
                print_color(RGB::from_f32(0.0, n, 0.0), "░");
            }
        }
        print_color(RGB::named(WHITE), "\n");
    }

    print_color(RGB::named(WHITE), "\n");
    stdout().flush().expect("flush failed");

    println!("{min}, {max}");

}
