use glam::Vec3A;

// Utility functions
pub(super) fn fast_floor(f: f32) -> i32 {
    f.floor() as _
}

pub(super) fn fast_round(f: f32) -> i32 {
    if f >= 0.0 {
        (f + 0.5) as i32
    } else {
        (f - 0.5) as i32
    }
}

#[allow(dead_code)]
pub(super) fn fast_abs(i: i32) -> i32 {
    i32::abs(i)
}

pub(super) fn fast_abs_f(i: f32) -> f32 {
    f32::abs(i)
}

pub(super) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

pub(super) fn interp_hermite_func(t: f32) -> f32 {
    t * t * (3. - 2. * t)
}
pub(super) fn interp_hermite_func_vec(t: Vec3A) -> Vec3A {
    t * t * (3.0 - 2.0 * t)
}

pub(super) fn interp_quintic_func(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
pub(super) fn interp_quintic_func_vec(t: Vec3A) -> Vec3A {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[allow(clippy::many_single_char_names)]
pub(super) fn cubic_lerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let p = (d - c) - (a - b);
    t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b
}