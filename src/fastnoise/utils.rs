use glam::{IVec3, Vec3A};

use crate::fastnoise::consts::{X_PRIME, Y_PRIME, Z_PRIME};

use super::consts::{GRAD_A, VAL_LUT};

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

#[inline]
pub(super) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

pub(super) fn interp_hermite_func(t: f32) -> f32 {
    t * t * (3. - 2. * t)
}

#[inline]
pub(super) fn interp_hermite_func_vec(t: Vec3A) -> Vec3A {
    t * t * (3.0 - 2.0 * t)
}

pub(super) fn interp_quintic_func(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
pub(super) fn interp_quintic_func_vec(t: Vec3A) -> Vec3A {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[allow(clippy::many_single_char_names)]
#[inline]
pub(super) fn cubic_lerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let q = a - b;
    let p = d - c - q;
    t * t * t * p + t * t * (q - p) + t * (c - a) + b
}

#[inline]
pub(super) fn grad_coord_3d(perm: &[u8], perm12: &[u8], offset: u8, a: IVec3, b: Vec3A) -> f32 {
    let lut_pos = index3d_12(perm, perm12, offset, a) as usize;
    (b * GRAD_A[lut_pos]).element_sum()
}

#[inline]
pub(super) fn index3d_12(perm: &[u8], perm12: &[u8], offset: u8, v: IVec3) -> u8 {
    let z = (v.z as usize & 0xFF) + offset as usize;
    let y = (v.y as usize & 0xFF) + perm[z] as usize;
    let x = (v.x as usize & 0xFF) + perm[y] as usize;
    perm12[x]
}

#[inline]
pub(super) fn val_coord_3d_fast(perm: &[u8], offset: u8, pos: IVec3) -> f32 {
    VAL_LUT[index3d_256(perm, offset, pos) as usize]
}

#[inline]
pub(super) fn val_coord_3d(seed: i32, pos: IVec3) -> f32 {
    use std::num::Wrapping;

    let mut n = Wrapping(seed);
    n ^= Wrapping(X_PRIME) * Wrapping(pos.x);
    n ^= Wrapping(Y_PRIME) * Wrapping(pos.y);
    n ^= Wrapping(Z_PRIME) * Wrapping(pos.z);

    (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
}

#[inline]
pub(super) fn index3d_256(perm: &[u8], offset: u8, pos: IVec3) -> u8 {
    let z = (pos.z as usize & 0xFF) + offset as usize;
    let y = (pos.y as usize & 0xFF) + perm[z] as usize;
    let x = (pos.x as usize & 0xFF) + perm[y] as usize;
    perm[x]
}