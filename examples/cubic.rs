use bracket_color::prelude::*;
use crossterm::queue;
use crossterm::style::{Color::Rgb, Print, SetForegroundColor};
use fastnoise::{Builder as _, Sampler as _, CubicNoiseBuilder};
use std::f32::consts::PI;
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
    let noise = CubicNoiseBuilder {
        // domain: Some([0.2, 13.0]),
        frequency: PI,
        seed: 31337,
        ..Default::default()
    }.build();

    let mut max = f32::MIN;
    let mut min = f32::MAX;

    for y in 0..50 {
        for x in 0..80 {
            let n = noise.sample2d([x as f32, y as f32]);

            max = max.max(n);
            min = min.min(n);

            print_color(RGB::from_f32(n, n, n), "▒");
        }
        print_color(RGB::named(WHITE), "\n");
    }

    print_color(RGB::named(WHITE), "\n");
    stdout().flush().expect("Flush Fail");

    println!("{min}, {max}");

}
