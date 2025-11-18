// A port of Auburn's FastNoise to Rust.
// I really didn't like the noise libraries I could find, so I ported the one I like.
// Original code: https://github.com/Auburns/FastNoise
// The original is MIT licensed, so this is compatible.

// Updated by Scott Will to use glam's SIMD enabled Vec3A as much as possible

use glam::*;
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

use super::consts::*;

#[derive(Debug, Default, PartialEq, Copy, Clone, Deserialize, Serialize)]
/// Type of noise to generate
pub enum NoiseType {
    Value,
    ValueFractal,
    Perlin,
    PerlinFractal,
    #[default]
    Simplex,
    SimplexFractal,
    Cellular,
    WhiteNoise,
    Cubic,
    CubicFractal,
}

impl Display for NoiseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone, Deserialize, Serialize)]
/// Interpolation function to use
pub enum Interp {
    Linear,
    Hermite,
    #[default]
    Quintic,
}

impl Display for Interp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone, Deserialize, Serialize)]
/// Fractal function to use
pub enum FractalType {
    #[default]
    FBM,
    Billow,
    RigidMulti,
}

impl Display for FractalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone, Deserialize, Serialize)]
/// Cellular noise distance function to use
pub enum CellularDistanceFunction {
    #[default]
    Euclidean,
    Manhattan,
    Natural,
}

impl Display for CellularDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Debug, Default, PartialEq, Copy, Clone, Deserialize, Serialize)]
/// What type of cellular noise result do you want
pub enum CellularReturnType {
    #[default]
    CellValue,
    Distance,
    Distance2,
    Distance2Add,
    Distance2Sub,
    Distance2Mul,
    Distance2Div,
}

impl Display for CellularReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct FastNoise {
    seed: u64,
    frequency: f32,
    interp: Interp,
    noise_type: NoiseType,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    fractal_type: FractalType,
    cellular_distance_function: CellularDistanceFunction,
    cellular_return_type: CellularReturnType,
    cellular_distance_index: (i32, i32),
    cellular_jitter: f32,
    gradient_perturb_amp: f32,
    perm: Vec<u8>,
    perm12: Vec<u8>,
    fractal_bounding: f32,
}

impl Default for FastNoise {
    fn default() -> Self {
        let mut noise = Self {
            seed: 0,
            frequency: 1.0,
            interp: Interp::default(),
            noise_type: NoiseType::default(),
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            fractal_type: FractalType::default(),
            cellular_distance_function: CellularDistanceFunction::default(),
            cellular_return_type: CellularReturnType::default(),
            cellular_distance_index: (0, 1),
            cellular_jitter: 0.5,
            gradient_perturb_amp: 1.0,
            perm: vec![0; 512],
            perm12: vec![0; 512],
            fractal_bounding: 0.0,
        };
        noise.set_seed(0);
        noise.calculate_fractal_bounding();
        noise
    }
}


// Utility functions
#[inline(always)]
fn extend(pos: Vec2) -> Vec3A {
    vec3a(pos.x, pos.y, 0.0)
}

#[inline(always)]
pub const fn ivec33(x: &IVec3, y: &IVec3, z: &IVec3) -> IVec3 {
    IVec3::new(x.x, y.y, z.z)
}

fn fast_floor(f: f32) -> i32 {
    if f >= 0.0 {
        f as i32
    } else {
        f as i32 - 1
    }
}

fn fast_round(f: f32) -> i32 {
    if f >= 0.0 {
        (f + 0.5) as i32
    } else {
        (f - 0.5) as i32
    }
}

#[allow(dead_code)]
fn fast_abs(i: i32) -> i32 {
    i32::abs(i)
}

fn fast_abs_f(i: f32) -> f32 {
    f32::abs(i)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn interp_hermite_func(t: Vec3A) -> Vec3A {
    t * t * (3.0 - 2.0 * t)
}

fn interp_quintic_func(t: Vec3A) -> Vec3A {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[allow(clippy::many_single_char_names)]
fn cubic_lerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let p = (d - c) - (a - b);
    t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b
}

#[allow(clippy::unreadable_literal)]
#[allow(clippy::new_without_default)]
impl FastNoise {

    /// Creates a new noise instance specifying a random seed.
    pub fn seeded(seed: u64) -> FastNoise {
        let mut noise = FastNoise { seed, ..Default::default() };
        noise.set_seed(seed);
        noise.calculate_fractal_bounding();
        noise
    }

    /// Re-seeds the noise system with a new seed.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;

        for i in 0..=255 {
            self.perm[i as usize] = i;
        }

        let mut rng: Pcg64 = Seeder::from(&seed).into_rng();
        for j in 0..256 {
            let rng = rng.random::<u64>() % (256 - j);
            let k = rng + j;
            let l = self.perm[j as usize];
            self.perm[j as usize] = self.perm[k as usize];
            self.perm[j as usize + 256] = self.perm[k as usize];
            self.perm[k as usize] = l;
            self.perm12[j as usize] = self.perm[j as usize] % 12;
            self.perm12[j as usize + 256] = self.perm[j as usize] % 12;
        }
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }
    pub fn get_frequency(&self) -> f32 {
        self.frequency
    }
    pub fn set_interp(&mut self, interp: Interp) {
        self.interp = interp;
    }
    pub fn get_interp(&self) -> Interp {
        self.interp
    }
    pub fn set_noise_type(&mut self, nt: NoiseType) {
        self.noise_type = nt;
    }
    pub fn get_noise_type(&self) -> NoiseType {
        self.noise_type
    }
    pub fn set_fractal_octaves(&mut self, octaves: i32) {
        self.octaves = octaves;
        self.calculate_fractal_bounding();
    }
    pub fn get_fractal_octaves(&self) -> i32 {
        self.octaves
    }
    pub fn set_fractal_lacunarity(&mut self, lacunarity: f32) {
        self.lacunarity = lacunarity;
    }
    pub fn get_fractal_lacunarity(&self) -> f32 {
        self.lacunarity
    }
    pub fn set_fractal_gain(&mut self, gain: f32) {
        self.gain = gain;
        self.calculate_fractal_bounding();
    }
    pub fn get_fractal_gain(&self) -> f32 {
        self.gain
    }
    pub fn set_fractal_type(&mut self, fractal_type: FractalType) {
        self.fractal_type = fractal_type;
    }
    pub fn get_fractal_type(&self) -> FractalType {
        self.fractal_type
    }
    pub fn set_cellular_distance_function(
        &mut self,
        cellular_distance_function: CellularDistanceFunction,
    ) {
        self.cellular_distance_function = cellular_distance_function;
    }
    pub fn get_cellular_distance_function(&self) -> CellularDistanceFunction {
        self.cellular_distance_function
    }
    pub fn set_cellular_return_type(&mut self, cellular_return_type: CellularReturnType) {
        self.cellular_return_type = cellular_return_type;
    }
    pub fn get_cellular_return_type(&self) -> CellularReturnType {
        self.cellular_return_type
    }
    pub fn get_cellular_distance_indices(&self) -> (i32, i32) {
        self.cellular_distance_index
    }
    pub fn set_cellular_jitter(&mut self, jitter: f32) {
        self.cellular_jitter = jitter;
    }
    pub fn get_cellular_jitter(&self) -> f32 {
        self.cellular_jitter
    }
    pub fn set_gradient_perterb_amp(&mut self, gradient_perturb_amp: f32) {
        self.gradient_perturb_amp = gradient_perturb_amp;
    }
    pub fn get_gradient_perterb_amp(&self) -> f32 {
        self.gradient_perturb_amp
    }

    fn calculate_fractal_bounding(&mut self) {
        let mut amp: f32 = self.gain;
        let mut amp_fractal: f32 = 1.0;
        for _ in 0..self.octaves {
            amp_fractal += amp;
            amp *= self.gain;
        }
        self.fractal_bounding = 1.0 / amp_fractal;
    }

    pub fn set_cellular_distance_indices(&mut self, ixs: IVec2) {
        self.cellular_distance_index.0 = ixs.min_element().clamp(0, FN_CELLULAR_INDEX_MAX as i32);
        self.cellular_distance_index.1 = ixs.max_element().clamp(0, FN_CELLULAR_INDEX_MAX as i32);
    }

    pub fn index2d_12(&self, offset: u8, pos: IVec2) -> u8 {
        let y = (pos.y & 0xff) as usize + offset as usize;
        let x = (pos.x & 0xff) as usize + self.perm[y] as usize;
        self.perm12[x]
    }

    pub fn index3d_12(&self, offset: u8, pos: IVec3) -> u8 {
        let z = (pos.z as usize & 0xff) + offset as usize;
        let y = (pos.y as usize & 0xff) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xff) + self.perm[y] as usize;
        self.perm12[x]
    }

    pub fn index4d_32(&self, offset: u8, pos: IVec4) -> u8 {
        let w = (pos.w as usize & 0xff) + offset as usize;
        let z = (pos.z as usize & 0xff) + self.perm[w] as usize;
        let y = (pos.y as usize & 0xff) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xff) + self.perm[y] as usize;
        self.perm[x] & 31
    }

    pub fn index2d_256(&self, offset: u8, pos: IVec2) -> u8 {
        let y = (pos.y as usize & 0xff) + offset as usize;
        let x = (pos.x as usize & 0xff) + self.perm[y] as usize;
        self.perm[x]
    }

    pub fn index3d_256(&self, offset: u8, pos: IVec3) -> u8 {
        let z = (pos.z as usize & 0xff) + offset as usize;
        let y = (pos.y as usize & 0xff) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xff) + self.perm[y] as usize;
        self.perm[x]
    }

    pub fn index4d_256(&self, offset: u8, pos: IVec4) -> u8 {
        let w = (pos.w as usize & 0xff) + offset as usize;
        let z = (pos.z as usize & 0xff) + self.perm[w] as usize;
        let y = (pos.y as usize & 0xff) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xff) + self.perm[y] as usize;
        self.perm[x]
    }

    fn val_coord_2d(&self, seed: i32, pos: IVec2) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(PRIME.x) * Wrapping(pos.x);
        n ^= Wrapping(PRIME.y) * Wrapping(pos.y);
        (n * n * n * Wrapping(COORD_A)).0 as f32 / COORD_B
    }

    fn val_coord_3d(&self, seed: i32, pos: IVec3,) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(PRIME.x) * Wrapping(pos.x);
        n ^= Wrapping(PRIME.y) * Wrapping(pos.y);
        n ^= Wrapping(PRIME.z) * Wrapping(pos.z);

        (n * n * n * Wrapping(COORD_A)).0 as f32 / COORD_B
    }

    #[allow(dead_code)]
    fn val_coord_4d(&self, seed: i32, pos: IVec4) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(PRIME.x) * Wrapping(pos.x);
        n ^= Wrapping(PRIME.y) * Wrapping(pos.y);
        n ^= Wrapping(PRIME.z) * Wrapping(pos.z);
        n ^= Wrapping(PRIME.w) * Wrapping(pos.w);

        (n * n * n * Wrapping(COORD_A)).0 as f32 / COORD_B
    }

    fn val_coord_2d_fast(&self, offset: u8, pos: IVec2) -> f32 {
        VAL_LUT[self.index2d_256(offset, pos) as usize]
    }
    fn val_coord_3d_fast(&self, offset: u8, pos: IVec3) -> f32 {
        VAL_LUT[self.index3d_256(offset, pos) as usize]
    }

    fn grad_coord_2d(&self, offset: u8, pos: IVec2, delta: Vec2) -> f32 {
        let lut_pos = self.index2d_12(offset, pos) as usize;
        let grad = GRAD[lut_pos];
        (delta * grad.xy()).element_sum()
    }

    fn grad_coord_3d(&self, offset: u8, pos: IVec3, delta: Vec3A) -> f32 {
        let lut_pos = self.index3d_12(offset, pos) as usize;
        let grad = GRAD[lut_pos];
        (delta * grad).element_sum()
    }

    // #[allow(dead_code)]
    // fn grad_coord_4d(&self, offset: u8, pos: IVec4, delta: Vec4) -> f32 {
    //     let lut_pos = self.index4d_32(offset, pos) as usize;
    //     let grad = vec4(
    //         GRAD_4D[lut_pos],
    //         GRAD_4D[lut_pos + 1],
    //         GRAD_4D[lut_pos + 2],
    //         GRAD_4D[lut_pos + 3],
    //     );
    //     (delta * grad).element_sum()
    // }

    pub fn get_noise3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.noise_type {
            NoiseType::Value => self.single_value3d(0, pos),
            NoiseType::ValueFractal => match self.fractal_type {
                FractalType::FBM => self.single_value_fractal_fbm3d(pos),
                FractalType::Billow => self.single_value_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(pos),
            },
            NoiseType::Perlin => self.single_perlin3d(0, pos),
            NoiseType::PerlinFractal => match self.fractal_type {
                FractalType::FBM => self.single_perlin_fractal_fbm3d(pos),
                FractalType::Billow => self.single_perlin_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d(pos),
            },
            NoiseType::Simplex => self.single_simplex3d(0, pos),
            NoiseType::SimplexFractal => match self.fractal_type {
                FractalType::FBM => self.single_simplex_fractal_fbm3d(pos),
                FractalType::Billow => self.single_simplex_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi3d(pos),
            },
            NoiseType::Cellular => match self.cellular_return_type {
                CellularReturnType::CellValue => self.single_cellular3d(pos),
                CellularReturnType::Distance => self.single_cellular3d(pos),
                _ => self.single_cellular_2edge3d(pos),
            },
            NoiseType::WhiteNoise => self.get_white_noise3d(pos),
            NoiseType::Cubic => self.single_cubic3d(0, pos),
            NoiseType::CubicFractal => match self.fractal_type {
                FractalType::FBM => self.single_cubic_fractal_fbm3d(pos),
                FractalType::Billow => self.single_cubic_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(pos),
            },
        }
    }

    pub fn get_noise(&self, mut pos: Vec2) -> f32 {
        self.get_noise3d(extend(pos))
    }

    #[allow(dead_code)]
    fn get_white_noise4d(&self, pos: Vec4) -> f32 {
        // this is doing it a bit differentky than `pos.as_ivec4()`
        let pos = ivec4(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
            pos.w.to_bits() as i32,
        );

        self.val_coord_4d(self.seed as i32, pos ^ (pos >> 16))
    }

    fn get_white_noise3d(&self, pos: Vec3A) -> f32 {
        // this is doing it a bit differentky than `pos.as_ivec3()`
        let pos = ivec3(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
        );

        self.val_coord_3d(self.seed as i32, pos ^ (pos >> 16))
    }

    fn get_white_noise(&self, pos: Vec2) -> f32 {
        // this is doing it a bit differentky than `pos.as_ivec2()`
        let pos = ivec2(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
        );

        self.val_coord_2d(self.seed as i32, pos ^ (pos >> 16))
    }

    #[allow(dead_code)]
    fn get_white_noise_int4d(&self, pos: IVec4) -> f32 {
        self.val_coord_4d(self.seed as i32, pos)
    }

    #[allow(dead_code)]
    fn get_white_noise_int3d(&self, pos: IVec3) -> f32 {
        self.val_coord_3d(self.seed as i32, pos)
    }

    #[allow(dead_code)]
    fn get_white_noise_int(&self, pos: IVec2) -> f32 {
        self.val_coord_2d(self.seed as i32, pos)
    }

    // Value noise

    #[allow(dead_code)]
    fn get_value_fractal3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_value_fractal_fbm3d(pos),
            FractalType::Billow => self.single_value_fractal_billow3d(pos),
            FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(pos),
        }
    }

    #[allow(dead_code)]
    fn get_value_fractal(&self, mut pos: Vec2) -> f32 {
        self.get_value3d(extend(pos))
    }

    fn single_value_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_value3d(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_value3d(self.perm[i as usize], pos) * amp;

            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = fast_abs_f(self.single_value3d(self.perm[0], pos)) * 2.0 - 1.0;
        let mut amp: f32 = 1.0;
        let mut i: i32 = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += (fast_abs_f(self.single_value3d(self.perm[i as usize], pos)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - fast_abs_f(self.single_value3d(self.perm[0], pos));
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_value3d(self.perm[i as usize], pos))) * amp;
            i += 1;
        }
        sum
    }

    #[allow(dead_code)]
    fn get_value3d(&self, pos: Vec3A) -> f32 {
        self.single_value3d(0, pos * self.frequency)
    }

    fn single_value3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor();

        let s: Vec3A = {
            let d = pos - p0;
            match self.interp {
                Interp::Linear => d,
                Interp::Hermite => interp_hermite_func(d),
                Interp::Quintic => interp_quintic_func(d),
            }
        };

        let p0 = p0.as_ivec3();
        let p1 = p0 + 1;

        let xf00: f32 = lerp(
            self.val_coord_3d_fast(offset, p0),
            self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p0.z)),
            s.x,
        );
        let xf10: f32 = lerp(
            self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p0.z)),
            self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p0.z)),
            s.x,
        );
        let xf01: f32 = lerp(
            self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p1.z)),
            self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p1.z)),
            s.x,
        );
        let xf11: f32 = lerp(
            self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p1.z)),
            self.val_coord_3d_fast(offset, p1),
            s.x,
        );

        let yf0: f32 = lerp(xf00, xf10, s.y);
        let yf1: f32 = lerp(xf01, xf11, s.y);

        lerp(yf0, yf1, s.z)
    }

    fn single_value_fractal_fbm(&self, mut pos: Vec2) -> f32 {
        self.single_value_fractal_fbm3d(extend(pos))
    }

    fn single_value_fractal_billow(&self, mut pos: Vec2) -> f32 {
        self.single_value_fractal_billow3d(extend(pos))
    }

    fn single_value_fractal_rigid_multi(&self, mut pos: Vec2) -> f32 {
        self.single_value_fractal_rigid_multi3d(extend(pos))
    }

    #[allow(dead_code)]
    fn get_value(&self, pos: Vec2) -> f32 {
        self.single_value(0, pos * self.frequency)
    }

    fn single_value(&self, offset: u8, pos: Vec2) -> f32 {
        self.single_value3d(offset, extend(pos))
    }

    // Perlin noise

    #[allow(dead_code)]
    fn get_perlin_fractal3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_perlin_fractal_fbm3d(pos),
            FractalType::Billow => self.single_perlin_fractal_billow3d(pos),
            FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d(pos),
        }
    }

    fn single_perlin_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_perlin3d(self.perm[0], pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = fast_abs_f(self.single_perlin3d(self.perm[0], pos)) * 2.0 - 1.0;
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += (fast_abs_f(self.single_perlin3d(self.perm[i as usize], pos)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - fast_abs_f(self.single_perlin3d(self.perm[0], pos));
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_perlin3d(self.perm[i as usize], pos))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_perlin3d(&self, pos: Vec3A) -> f32 {
        self.single_perlin3d(0, pos * self.frequency)
    }

    fn single_perlin3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor();

        let s = {
            let d = pos - p0;
            match self.interp {
                Interp::Linear => d,
                Interp::Hermite => interp_hermite_func(d),
                Interp::Quintic => interp_quintic_func(d),
            }
        };

        let d0 = pos - p0;
        let d1 = d0 - 1.0;

        let p0 = p0.as_ivec3();
        let p1 = p0 + 1;

        let xf00 = lerp(
            self.grad_coord_3d(offset, p0, d0),
            self.grad_coord_3d(offset, ivec3(p1.x, p0.y, p0.z), vec3a(d1.x, d0.y, d0.z)),
            s.x,
        );
        let xf10 = lerp(
            self.grad_coord_3d(offset, ivec3(p0.x, p1.y, p0.z), vec3a(d0.x, d1.y, d0.z)),
            self.grad_coord_3d(offset, ivec3(p1.x, p1.y, p0.z), vec3a(d1.x, d1.y, d0.z)),
            s.x,
        );
        let xf01 = lerp(
            self.grad_coord_3d(offset, ivec3(p0.x, p0.y, p1.z), vec3a(d0.x, d0.y, d1.z)),
            self.grad_coord_3d(offset, ivec3(p1.x, p0.y, p1.z), vec3a(d1.x, d0.y, d1.z)),
            s.x,
        );
        let xf11 = lerp(
            self.grad_coord_3d(offset, ivec3(p0.x, p1.y, p1.z), vec3a(d0.x, d1.y, d1.z)),
            self.grad_coord_3d(offset, p1, d1),
            s.x,
        );

        let yf0 = lerp(xf00, xf10, s.y);
        let yf1 = lerp(xf01, xf11, s.y);

        lerp(yf0, yf1, s.z)
    }

    #[allow(dead_code)]
    fn get_perlin_fractal(&self, pos: Vec2) -> f32 {
        self.get_perlin_fractal3d(extend(pos))
    }

    fn single_perlin_fractal_fbm(&self, pos: Vec2) -> f32 {
        self.single_perlin_fractal_fbm3d(extend(pos))
    }

    fn single_perlin_fractal_billow(&self, pos: Vec2) -> f32 {
        self.single_perlin_fractal_billow3d(extend(pos))
    }

    fn single_perlin_fractal_rigid_multi(&self, pos: Vec2) -> f32 {
        self.single_perlin_fractal_rigid_multi3d(extend(pos))
    }

    #[allow(dead_code)]
    fn get_perlin(&self, pos: Vec2) -> f32 {
        self.single_perlin(0, pos * self.frequency)
    }

    fn single_perlin(&self, offset: u8, pos: Vec2) -> f32 {
        self.single_perlin3d(offset, extend(pos))
    }

    #[allow(dead_code)]
    // Simplex noise
    fn get_simplex_fractal3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_simplex_fractal_fbm3d(pos),
            FractalType::Billow => self.single_simplex_fractal_billow3d(pos),
            FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi3d(pos),
        }
    }

    fn single_simplex_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_simplex3d(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_simplex3d(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = fast_abs_f(self.single_simplex3d(self.perm[0], pos)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += (fast_abs_f(self.single_simplex3d(self.perm[i as usize], pos)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_simplex3d(self.perm[0], pos));
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_simplex3d(self.perm[i as usize], pos))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_simplex3d(&self, pos: Vec3A) -> f32 {
        self.single_simplex3d(0, pos * self.frequency)
    }

    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::collapsible_if)]
    #[allow(clippy::suspicious_else_formatting)]
    fn single_simplex3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let mut t = pos.element_sum() * F3;
        let q = (pos + t).floor();

        t = q.element_sum() * G3;
        let p0 = pos - q - t;

        let q1: Vec3A;
        let q2: Vec3A;

        if p0.x >= p0.y {
            if p0.y >= p0.z {
                q1 = vec3a(1.0, 0.0, 0.0);
                q2 = vec3a(1.0, 1.0, 0.0);
            } else if p0.x >= p0.z {
                q1 = vec3a(1.0, 0.0, 0.0);
                q2 = vec3a(1.0, 0.0, 1.0);
            } else
            // x0 < z0
            {
                q1 = vec3a(0.0, 0.0, 1.0);
                q2 = vec3a(1.0, 0.0, 1.0);
            }
        } else
        // x0 < y0
        {
            if p0.y < p0.z {
                q1 = vec3a(0.0, 0.0, 1.0);
                q2 = vec3a(0.0, 1.0, 1.0);
            } else if p0.x < p0.z {
                q1 = vec3a(0.0, 1.0, 0.0);
                q2 = vec3a(0.0, 1.0, 1.0);
            } else
            // x0 >= z0
            {
                q1 = vec3a(0.0, 1.0, 0.0);
                q2 = vec3a(1.0, 1.0, 0.0);
            }
        }

        let p1 = p0 - q1 + G3;
        let p2 = p0 - q2 + 2.0 * G3;
        let p3 = p0 - 1.0 + 3.0 * G3;

        let n0;
        let n1;
        let n2;
        let n3;

        let p00 = p0 * p0;
        t = 0.6 - p00.x - p00.y - p00.z;
        if t < 0.0 {
            n0 = 0.0;
        } else {
            t *= t;
            n0 = t * t * self.grad_coord_3d(offset, q.as_ivec3(), p0);
        }

        let p11 = p1 * p1;
        t = 0.6 - p11.x - p11.y - p11.z;
        if t < 0. {
            n1 = 0.
        } else {
            t *= t;
            n1 = t
                * t
                * self.grad_coord_3d(offset, (q + q1).as_ivec3(), p1);
        }

        let p22 = p2 * p2;
        t = 0.6 - p22.x - p22.y - p22.z;
        if t < 0. {
            n2 = 0.
        } else {
            t *= t;
            n2 = t
                * t
                * self.grad_coord_3d(offset, (q + q2).as_ivec3(), p2);
        }

        let p33 = p3 * p3;
        t = 0.6 - p33.x - p33.y - p33.z;
        if t < 0. {
            n3 = 0.
        } else {
            t *= t;
            n3 = t * t * self.grad_coord_3d(offset, q.as_ivec3() + 1, p3);
        }

        32.0 * (n0 + n1 + n2 + n3)
    }

    #[allow(dead_code)]
    fn get_simplex_fractal(&self, pos: Vec2) -> f32 {
        self.get_simplex_fractal3d(extend(pos))
    }

    fn single_simplex_fractal_fbm(&self, pos: Vec2) -> f32 {
        self.single_simplex_fractal_fbm3d(extend(pos))
    }

    fn single_simplex_fractal_billow(&self, pos: Vec2) -> f32 {
        self.single_simplex_fractal_billow3d(extend(pos))
    }

    fn single_simplex_fractal_rigid_multi(&self, pos: Vec2) -> f32 {
        self.single_simplex_fractal_rigid_multi3d(extend(pos))
    }

    // #[allow(dead_code)]
    // fn single_simplex_fractal_blend(&self, mut x: f32, mut y: f32) -> f32 {
    //     let mut sum = self.single_simplex(self.perm[0], x, y);
    //     let mut amp = 1.0;
    //     let mut i = 1;

    //     while i < self.octaves {
    //         x *= self.lacunarity;
    //         y *= self.lacunarity;

    //         amp *= self.gain;
    //         sum += self.single_simplex(self.perm[i as usize], x, y) * amp + 1.0;
    //         i += 1;
    //     }

    //     sum * self.fractal_bounding
    // }

    #[allow(dead_code)]
    fn get_simplex(&self, pos: Vec2) -> f32 {
        self.single_simplex(0, pos * self.frequency)
    }

    #[allow(clippy::many_single_char_names)]
    fn single_simplex(&self, offset: u8, pos: Vec2) -> f32 {
        self.single_simplex3d(offset, extend(pos))
        // let mut t: f32 = (x + y) * F2;
        // let i = fast_floor(x + t);
        // let j = fast_floor(y + t);

        // t = (i + j) as f32 * G2;
        // let x0 = i as f32 - t;
        // let y0 = j as f32 - t;

        // let x0 = x - x0;
        // let y0 = y - y0;

        // let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        // let x1 = x0 - i1 as f32 + G2;
        // let y1 = y0 - j1 as f32 + G2;
        // let x2 = x0 - 1.0 + 2.0 * G2;
        // let y2 = y0 - 1.0 + 2.0 * G2;

        // let n0;
        // let n1;
        // let n2;

        // t = 0.5 - x0 * x0 - y0 * y0;
        // if t < 0. {
        //     n0 = 0.
        // } else {
        //     t *= t;
        //     n0 = t * t * self.grad_coord_2d(offset, i, j, x0, y0);
        // }

        // t = 0.5 - x1 * x1 - y1 * y1;
        // if t < 0. {
        //     n1 = 0.
        // } else {
        //     t *= t;
        //     n1 = t * t * self.grad_coord_2d(offset, i + i1 as i32, j + j1 as i32, x1, y1);
        // }

        // t = 0.5 - x2 * x2 - y2 * y2;
        // if t < 0. {
        //     n2 = 0.
        // } else {
        //     t *= t;
        //     n2 = t * t * self.grad_coord_2d(offset, i + 1, j + 1, x2, y2);
        // }

        // 70.0 * (n0 + n1 + n2)
    }

    // #[allow(dead_code)]
    // fn get_simplex_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 {
    //     self.single_simplex4d(
    //         0,
    //         x * self.frequency,
    //         y * self.frequency,
    //         z * self.frequency,
    //         w * self.frequency,
    //     )
    // }

    #[allow(dead_code)]
    fn greater_1_0(&self, n: i32, greater_than: i32) -> i32 {
        if n >= greater_than {
            1
        } else {
            0
        }
    }

    // #[allow(dead_code)]
    // #[allow(clippy::many_single_char_names)]
    // fn single_simplex4d(&self, offset: u8, x: f32, y: f32, z: f32, w: f32) -> f32 {
    //     let n0: f32;
    //     let n1: f32;
    //     let n2: f32;
    //     let n3: f32;
    //     let n4: f32;

    //     let mut t = (x + y + z + w) * F4;
    //     let i = fast_floor(x + t) as f32;
    //     let j = fast_floor(y + t) as f32;
    //     let k = fast_floor(z + t) as f32;
    //     let l = fast_floor(w + t) as f32;
    //     t = (i + j + k + l) * G4;
    //     let x0 = i - t;
    //     let y0 = j - t;
    //     let z0 = k - t;
    //     let w0 = l - t;
    //     let x0 = x - x0;
    //     let y0 = y - y0;
    //     let z0 = z - z0;
    //     let w0 = w - w0;

    //     let mut rankx = 0;
    //     let mut ranky = 0;
    //     let mut rankz = 0;
    //     let mut rankw = 0;

    //     if x0 > y0 {
    //         rankx += 1;
    //     } else {
    //         ranky += 1;
    //     }
    //     if x0 > z0 {
    //         rankx += 1;
    //     } else {
    //         rankz += 1
    //     };
    //     if x0 > w0 {
    //         rankx += 1;
    //     } else {
    //         rankw += 1
    //     };
    //     if y0 > z0 {
    //         ranky += 1;
    //     } else {
    //         rankz += 1
    //     };
    //     if y0 > w0 {
    //         ranky += 1;
    //     } else {
    //         rankw += 1
    //     };
    //     if z0 > w0 {
    //         rankz += 1;
    //     } else {
    //         rankw += 1
    //     };

    //     let i1 = self.greater_1_0(rankx, 3);
    //     let j1 = self.greater_1_0(ranky, 3);
    //     let k1 = self.greater_1_0(rankz, 3);
    //     let l1 = self.greater_1_0(rankw, 3);

    //     let i2 = self.greater_1_0(rankx, 2);
    //     let j2 = self.greater_1_0(ranky, 2);
    //     let k2 = self.greater_1_0(rankz, 2);
    //     let l2 = self.greater_1_0(rankw, 2);

    //     let i3 = self.greater_1_0(rankx, 1);
    //     let j3 = self.greater_1_0(ranky, 1);
    //     let k3 = self.greater_1_0(rankz, 1);
    //     let l3 = self.greater_1_0(rankw, 1);

    //     let x1 = x0 - i1 as f32 + G4;
    //     let y1 = y0 - j1 as f32 + G4;
    //     let z1 = z0 - k1 as f32 + G4;
    //     let w1 = w0 - l1 as f32 + G4;
    //     let x2 = x0 - i2 as f32 + 2.0 * G4;
    //     let y2 = y0 - j2 as f32 + 2.0 * G4;
    //     let z2 = z0 - k2 as f32 + 2.0 * G4;
    //     let w2 = w0 - l2 as f32 + 2.0 * G4;
    //     let x3 = x0 - i3 as f32 + 3.0 * G4;
    //     let y3 = y0 - j3 as f32 + 3.0 * G4;
    //     let z3 = z0 - k3 as f32 + 3.0 * G4;
    //     let w3 = w0 - l3 as f32 + 3.0 * G4;
    //     let x4 = x0 - 1.0 + 4.0 * G4;
    //     let y4 = y0 - 1.0 + 4.0 * G4;
    //     let z4 = z0 - 1.0 + 4.0 * G4;
    //     let w4 = w0 - 1.0 + 4.0 * G4;

    //     t = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
    //     if t < 0.0 {
    //         n0 = 0.;
    //     } else {
    //         t *= t;
    //         n0 = t
    //             * t
    //             * self.grad_coord_4d(
    //                 offset, i as i32, j as i32, k as i32, l as i32, x0, y0, z0, w0,
    //             );
    //     }
    //     t = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
    //     if t < 0.0 {
    //         n1 = 0.;
    //     } else {
    //         t *= t;
    //         n1 = t
    //             * t
    //             * self.grad_coord_4d(
    //                 offset,
    //                 i as i32 + i1,
    //                 j as i32 + j1,
    //                 k as i32 + k1,
    //                 l as i32 + l1,
    //                 x1,
    //                 y1,
    //                 z1,
    //                 w1,
    //             );
    //     }
    //     t = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
    //     if t < 0.0 {
    //         n2 = 0.;
    //     } else {
    //         t *= t;
    //         n2 = t
    //             * t
    //             * self.grad_coord_4d(
    //                 offset,
    //                 i as i32 + i2,
    //                 j as i32 + j2,
    //                 k as i32 + k2,
    //                 l as i32 + l2,
    //                 x2,
    //                 y2,
    //                 z2,
    //                 w2,
    //             );
    //     }
    //     t = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
    //     if t < 0.0 {
    //         n3 = 0.;
    //     } else {
    //         t *= t;
    //         n3 = t
    //             * t
    //             * self.grad_coord_4d(
    //                 offset,
    //                 i as i32 + i3,
    //                 j as i32 + j3,
    //                 k as i32 + k3,
    //                 l as i32 + l3,
    //                 x3,
    //                 y3,
    //                 z3,
    //                 w3,
    //             );
    //     }
    //     t = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
    //     if t < 0.0 {
    //         n4 = 0.;
    //     } else {
    //         t *= t;
    //         n4 = t
    //             * t
    //             * self.grad_coord_4d(
    //                 offset,
    //                 i as i32 + 1,
    //                 j as i32 + 1,
    //                 k as i32 + 1,
    //                 l as i32 + 1,
    //                 x4,
    //                 y4,
    //                 z4,
    //                 w4,
    //             );
    //     }

    //     27.0 * (n0 + n1 + n2 + n3 + n4) as f32
    // }

    #[allow(dead_code)]
    // Cubic Noise
    fn get_cubic_fractal3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_cubic_fractal_fbm3d(pos),
            FractalType::Billow => self.single_cubic_fractal_billow3d(pos),
            FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(pos),
        }
    }

    fn single_cubic_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_cubic3d(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_cubic3d(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = fast_abs_f(self.single_cubic3d(self.perm[0], pos)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum +=
                (fast_abs_f(self.single_cubic3d(self.perm[i as usize], pos)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_cubic3d(self.perm[0], pos));
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_cubic3d(self.perm[i as usize], pos))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_cubic3d(&self, pos: Vec3A) -> f32 {
        self.single_cubic3d(0, pos * self.frequency)
    }

    fn single_cubic3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p1 = pos.floor().as_ivec3();
        let p0 = p1 - 1;
        let p2 = p1 + 1;
        let p3 = p1 + 2;
        let ps = pos - p1.as_vec3a();

        cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p0, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p0, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p0, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p0, &p0)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p1, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p1, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p1, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p1, &p0)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p2, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p2, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p2, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p2, &p0)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p3, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p3, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p3, &p0)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p3, &p0)),
                    ps.x,
                ),
                ps.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p0, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p0, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p0, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p0, &p1)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p1, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p1, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p1, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p1, &p1)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p2, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p2, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p2, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p2, &p1)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p3, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p3, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p3, &p1)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p3, &p1)),
                    ps.x,
                ),
                ps.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p0, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p0, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p0, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p0, &p2)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p1, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p1, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p1, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p1, &p2)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p2, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p2, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p2, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p2, &p2)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p3, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p3, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p3, &p2)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p3, &p2)),
                    ps.x,
                ),
                ps.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p0, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p0, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p0, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p0, &p3)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p1, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p1, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p1, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p1, &p3)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p2, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p2, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p2, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p2, &p3)),
                    ps.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec33(&p0, &p3, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p1, &p3, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p2, &p3, &p3)),
                    self.val_coord_3d_fast(offset, ivec33(&p3, &p3, &p3)),
                    ps.x,
                ),
                ps.y,
            ),
            ps.z,
        ) * CUBIC_3D_BOUNDING
    }

    #[allow(dead_code)]
    fn get_cubic_fractal(&self, pos: Vec2) -> f32 {
        self.get_cubic_fractal3d(extend(pos))
    }

    fn single_cubic_fractal_fbm(&self, pos: Vec2) -> f32 {
        self.single_cubic_fractal_fbm3d(extend(pos))
    }

    fn single_cubic_fractal_billow(&self, pos: Vec2) -> f32 {
        self.single_cubic_fractal_billow3d(extend(pos))
    }

    fn single_cubic_fractal_rigid_multi(&self, pos: Vec2) -> f32 {
        self.single_cubic_fractal_rigid_multi3d(extend(pos))
    }

    #[allow(dead_code)]
    fn get_cubic(&self, pos: Vec2) -> f32 {
        // self.single_cubic(0, x * self.frequency, y * self.frequency)
        self.single_cubic(0, pos * self.frequency)
    }

    fn single_cubic(&self, offset: u8, pos: Vec2) -> f32 {
        self.single_cubic3d(offset, extend(pos))
        // let x1 = fast_floor(x);
        // let y1 = fast_floor(y);

        // let x0 = x1 - 1;
        // let y0 = y1 - 1;
        // let x2 = x1 + 1;
        // let y2 = y1 + 1;
        // let x3 = x1 + 2;
        // let y3 = y1 + 2;

        // let xs = x - x1 as f32;
        // let ys = y - y1 as f32;

        // cubic_lerp(
        //     cubic_lerp(
        //         self.val_coord_2d_fast(offset, x0, y0),
        //         self.val_coord_2d_fast(offset, x1, y0),
        //         self.val_coord_2d_fast(offset, x2, y0),
        //         self.val_coord_2d_fast(offset, x3, y0),
        //         xs,
        //     ),
        //     cubic_lerp(
        //         self.val_coord_2d_fast(offset, x0, y1),
        //         self.val_coord_2d_fast(offset, x1, y1),
        //         self.val_coord_2d_fast(offset, x2, y1),
        //         self.val_coord_2d_fast(offset, x3, y1),
        //         xs,
        //     ),
        //     cubic_lerp(
        //         self.val_coord_2d_fast(offset, x0, y2),
        //         self.val_coord_2d_fast(offset, x1, y2),
        //         self.val_coord_2d_fast(offset, x2, y2),
        //         self.val_coord_2d_fast(offset, x3, y2),
        //         xs,
        //     ),
        //     cubic_lerp(
        //         self.val_coord_2d_fast(offset, x0, y3),
        //         self.val_coord_2d_fast(offset, x1, y3),
        //         self.val_coord_2d_fast(offset, x2, y3),
        //         self.val_coord_2d_fast(offset, x3, y3),
        //         xs,
        //     ),
        //     ys,
        // ) * CUBIC_2D_BOUNDING
    }

    #[allow(dead_code)]
    // Cellular Noise
    fn get_cellular3d(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular3d(pos),
            CellularReturnType::Distance => self.single_cellular3d(pos),
            _ => self.single_cellular_2edge3d(pos),
        }
    }

    fn single_cellular3d(&self, pos: Vec3A) -> f32 {

        let [x, y, z] = pos.to_array();

        let xr = fast_round(x);
        let yr = fast_round(y);
        let zr = fast_round(z);

        let mut distance: f32 = 999999.0;
        let mut xc: i32 = 0;
        let mut yc: i32 = 0;
        let mut zc: i32 = 0;

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos: u8 = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

                            if new_distance < distance {
                                distance = new_distance;
                                xc = xi;
                                yc = yi;
                                zc = zi;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos: u8 = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance =
                                fast_abs_f(vec_x) + fast_abs_f(vec_y) + fast_abs_f(vec_z);

                            if new_distance < distance {
                                distance = new_distance;
                                xc = xi;
                                yc = yi;
                                zc = zi;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos: u8 = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance =
                                (fast_abs_f(vec_x) + fast_abs_f(vec_y) + fast_abs_f(vec_z))
                                    + (vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

                            if new_distance < distance {
                                distance = new_distance;
                                xc = xi;
                                yc = yi;
                                zc = zi;
                            }
                        }
                    }
                }
            }
        }

        //let lut_pos : u8;
        match self.cellular_return_type {
            CellularReturnType::CellValue => self.val_coord_3d(self.seed as i32, ivec3(xc, yc, zc)),
            CellularReturnType::Distance => distance,
            _ => 0.0,
        }
    }

    fn single_cellular_2edge3d(&self, pos: Vec3A) -> f32 {

        let [x, y, z] = pos.to_array();

        let xr = fast_round(x);
        let yr = fast_round(y);
        let zr = fast_round(z);

        let mut distance: Vec<f32> = vec![999999.0; FN_CELLULAR_INDEX_MAX as usize + 1];
        //FN_DECIMAL distance[FN_CELLULAR_INDEX_MAX+1] = { 999999,999999,999999,999999 };

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos: u8 = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

                            for i in (0..self.cellular_distance_index.1).rev() {
                                distance[i as usize] = f32::max(
                                    f32::min(distance[i as usize], new_distance),
                                    distance[i as usize - 1],
                                );
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance =
                                fast_abs_f(vec_x) + fast_abs_f(vec_y) + fast_abs_f(vec_z);

                            for i in (0..=self.cellular_distance_index.1).rev() {
                                distance[i as usize] = f32::max(
                                    f32::min(distance[i as usize], new_distance),
                                    distance[i as usize - 1],
                                );
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos = self.index3d_256(0, ivec3(xi, yi, zi));

                            let vec_x =
                                xi as f32 - x + CELL_3D_X[lut_pos as usize] * self.cellular_jitter;
                            let vec_y =
                                yi as f32 - y + CELL_3D_Y[lut_pos as usize] * self.cellular_jitter;
                            let vec_z =
                                zi as f32 - z + CELL_3D_Z[lut_pos as usize] * self.cellular_jitter;

                            let new_distance =
                                (fast_abs_f(vec_x) + fast_abs_f(vec_y) + fast_abs_f(vec_z))
                                    + (vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

                            for i in (0..=self.cellular_distance_index.1).rev() {
                                distance[i as usize] = f32::max(
                                    f32::min(distance[i as usize], new_distance),
                                    distance[i as usize - 1],
                                );
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
        }

        match self.cellular_return_type {
            CellularReturnType::Distance2 => distance[self.cellular_distance_index.1 as usize],
            CellularReturnType::Distance2Add => {
                distance[self.cellular_distance_index.1 as usize]
                    + distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Sub => {
                distance[self.cellular_distance_index.1 as usize]
                    - distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Mul => {
                distance[self.cellular_distance_index.1 as usize]
                    * distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Div => {
                distance[self.cellular_distance_index.0 as usize]
                    / distance[self.cellular_distance_index.1 as usize]
            }
            _ => 0.0,
        }
    }

    #[allow(dead_code)]
    fn get_cellular(&self, pos: Vec2) -> f32 {
        self.get_cellular3d(extend(pos))
    }

    fn single_cellular(&self, x: f32, y: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);

        let mut distance: f32 = 999999.0;

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = vec_x * vec_x + vec_y * vec_y;

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = fast_abs_f(vec_x) + fast_abs_f(vec_y);

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = (fast_abs_f(vec_x) + fast_abs_f(vec_y))
                            + (vec_x * vec_x + vec_y * vec_y);

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
        }

        //let lut_pos : u8;
        match self.cellular_return_type {
            CellularReturnType::CellValue => {
                self.val_coord_2d(self.seed as i32, ivec2(x as i32, y as i32))
            }
            _ => 0.0,
        }
    }

    fn single_cellular_2edge(&self, x: f32, y: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);

        let mut distance: Vec<f32> = vec![999999.0; FN_CELLULAR_INDEX_MAX as usize + 1];

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = vec_x * vec_x + vec_y * vec_y;

                        for i in (0..=self.cellular_distance_index.1).rev() {
                            distance[i as usize] = f32::max(
                                f32::min(distance[i as usize], new_distance),
                                distance[i as usize - 1],
                            );
                        }
                        distance[0] = f32::min(distance[0], new_distance);
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = fast_abs_f(vec_x) + fast_abs_f(vec_y);

                        for i in (0..=self.cellular_distance_index.1).rev() {
                            distance[i as usize] = f32::max(
                                f32::min(distance[i as usize], new_distance),
                                distance[i as usize - 1],
                            );
                        }
                        distance[0] = f32::min(distance[0], new_distance);
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos = self.index2d_256(0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = (fast_abs_f(vec_x) + fast_abs_f(vec_y))
                            + (vec_x * vec_x + vec_y * vec_y);

                        for i in (0..=self.cellular_distance_index.1).rev() {
                            distance[i as usize] = f32::max(
                                f32::min(distance[i as usize], new_distance),
                                distance[i as usize - 1],
                            );
                        }
                        distance[0] = f32::min(distance[0], new_distance);
                    }
                }
            }
        }

        match self.cellular_return_type {
            CellularReturnType::Distance2 => distance[self.cellular_distance_index.0 as usize],
            CellularReturnType::Distance2Add => {
                distance[self.cellular_distance_index.1 as usize]
                    + distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Sub => {
                distance[self.cellular_distance_index.1 as usize]
                    - distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Mul => {
                distance[self.cellular_distance_index.1 as usize]
                    * distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Div => {
                distance[self.cellular_distance_index.0 as usize]
                    / distance[self.cellular_distance_index.1 as usize]
            }
            _ => 0.0,
        }
    }

    // #[allow(dead_code)]
    // fn gradient_perturb3d(&self, pos: Vec3A) {
    //     self.single_gradient_perturb3d(0, self.gradient_perturb_amp, self.frequency, pos);
    // }

    // #[allow(dead_code)]
    // fn gradient_perturb_fractal3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
    //     let mut amp = self.gradient_perturb_amp * self.fractal_bounding;
    //     let mut freq = self.frequency;
    //     let mut i = 1;

    //     self.single_gradient_perturb3d(self.perm[0], amp, self.frequency, x, y, z);

    //     while i < self.octaves {
    //         freq *= self.lacunarity;
    //         amp *= self.gain;
    //         self.single_gradient_perturb3d(self.perm[i as usize], amp, freq, x, y, z);

    //         i += 1;
    //     }
    // }

    // #[allow(dead_code)]
    // fn single_gradient_perturb3d(
    //     &self,
    //     offset: u8,
    //     warp_amp: f32,
    //     frequency: f32,
    //     x: &mut f32,
    //     y: &mut f32,
    //     z: &mut f32,
    // ) {
    //     let xf = *x * frequency;
    //     let yf = *y * frequency;
    //     let zf = *z * frequency;

    //     let x0 = fast_floor(xf);
    //     let y0 = fast_floor(yf);
    //     let z0 = fast_floor(zf);
    //     let x1 = x0 + 1;
    //     let y1 = y0 + 1;
    //     let z1 = z0 + 1;

    //     let xs: f32;
    //     let ys: f32;
    //     let zs: f32;
    //     match self.interp {
    //         Interp::Linear => {
    //             xs = xf - x0 as f32;
    //             ys = yf - y0 as f32;
    //             zs = zf - z0 as f32;
    //         }
    //         Interp::Hermite => {
    //             xs = interp_hermite_func(xf - x0 as f32);
    //             ys = interp_hermite_func(yf - y0 as f32);
    //             zs = interp_hermite_func(zf - z0 as f32);
    //         }
    //         Interp::Quintic => {
    //             xs = interp_quintic_func(xf - x0 as f32);
    //             ys = interp_quintic_func(yf - y0 as f32);
    //             zs = interp_quintic_func(zf - z0 as f32);
    //         }
    //     }

    //     let mut lut_pos0 = self.index3d_256(offset, x0, y0, z0);
    //     let mut lut_pos1 = self.index3d_256(offset, x1, y0, z0);

    //     let mut lx0x = lerp(
    //         CELL_3D_X[lut_pos0 as usize],
    //         CELL_3D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     let mut ly0x = lerp(
    //         CELL_3D_Y[lut_pos0 as usize],
    //         CELL_3D_Y[lut_pos1 as usize],
    //         xs,
    //     );
    //     let mut lz0x = lerp(
    //         CELL_3D_Z[lut_pos0 as usize],
    //         CELL_3D_Z[lut_pos1 as usize],
    //         xs,
    //     );

    //     lut_pos0 = self.index3d_256(offset, x0, y1, z0);
    //     lut_pos1 = self.index3d_256(offset, x1, y1, z0);

    //     let mut lx1x = lerp(
    //         CELL_3D_X[lut_pos0 as usize],
    //         CELL_3D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     let mut ly1x = lerp(
    //         CELL_3D_Y[lut_pos0 as usize],
    //         CELL_3D_Y[lut_pos1 as usize],
    //         xs,
    //     );
    //     let mut lz1x = lerp(
    //         CELL_3D_Z[lut_pos0 as usize],
    //         CELL_3D_Z[lut_pos1 as usize],
    //         xs,
    //     );

    //     let lx0y = lerp(lx0x, lx1x, ys);
    //     let ly0y = lerp(ly0x, ly1x, ys);
    //     let lz0y = lerp(lz0x, lz1x, ys);

    //     lut_pos0 = self.index3d_256(offset, x0, y0, z1);
    //     lut_pos1 = self.index3d_256(offset, x1, y0, z1);

    //     lx0x = lerp(
    //         CELL_3D_X[lut_pos0 as usize],
    //         CELL_3D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     ly0x = lerp(
    //         CELL_3D_Y[lut_pos0 as usize],
    //         CELL_3D_Y[lut_pos1 as usize],
    //         xs,
    //     );
    //     lz0x = lerp(
    //         CELL_3D_Z[lut_pos0 as usize],
    //         CELL_3D_Z[lut_pos1 as usize],
    //         xs,
    //     );

    //     lut_pos0 = self.index3d_256(offset, x0, y1, z1);
    //     lut_pos1 = self.index3d_256(offset, x1, y1, z1);

    //     lx1x = lerp(
    //         CELL_3D_X[lut_pos0 as usize],
    //         CELL_3D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     ly1x = lerp(
    //         CELL_3D_Y[lut_pos0 as usize],
    //         CELL_3D_Y[lut_pos1 as usize],
    //         xs,
    //     );
    //     lz1x = lerp(
    //         CELL_3D_Z[lut_pos0 as usize],
    //         CELL_3D_Z[lut_pos1 as usize],
    //         xs,
    //     );

    //     *x += lerp(lx0y, lerp(lx0x, lx1x, ys), zs) * warp_amp;
    //     *y += lerp(ly0y, lerp(ly0x, ly1x, ys), zs) * warp_amp;
    //     *z += lerp(lz0y, lerp(lz0x, lz1x, ys), zs) * warp_amp;
    // }

    // #[allow(dead_code)]
    // fn gradient_perturb(&self, x: &mut f32, y: &mut f32) {
    //     self.single_gradient_perturb(0, self.gradient_perturb_amp, self.frequency, x, y);
    // }

    // #[allow(dead_code)]
    // fn gradient_perturb_fractal(&self, x: &mut f32, y: &mut f32) {
    //     let mut amp = self.gradient_perturb_amp * self.fractal_bounding;
    //     let mut freq = self.frequency;
    //     let mut i = 1;

    //     self.single_gradient_perturb(self.perm[0], amp, self.frequency, x, y);

    //     while i < self.octaves {
    //         freq *= self.lacunarity;
    //         amp *= self.gain;
    //         self.single_gradient_perturb(self.perm[i as usize], amp, freq, x, y);
    //         i += 1;
    //     }
    // }

    // #[allow(dead_code)]
    // fn single_gradient_perturb(
    //     &self,
    //     offset: u8,
    //     warp_amp: f32,
    //     frequency: f32,
    //     x: &mut f32,
    //     y: &mut f32,
    // ) {
    //     let xf = *x * frequency;
    //     let yf = *y * frequency;

    //     let x0 = fast_floor(xf);
    //     let y0 = fast_floor(yf);
    //     let x1 = x0 + 1;
    //     let y1 = y0 + 1;

    //     let xs: f32;
    //     let ys: f32;
    //     match self.interp {
    //         Interp::Linear => {
    //             xs = xf - x0 as f32;
    //             ys = yf - y0 as f32;
    //         }
    //         Interp::Hermite => {
    //             xs = interp_hermite_func(xf - x0 as f32);
    //             ys = interp_hermite_func(yf - y0 as f32);
    //         }
    //         Interp::Quintic => {
    //             xs = interp_quintic_func(xf - x0 as f32);
    //             ys = interp_quintic_func(yf - y0 as f32);
    //         }
    //     }

    //     let mut lut_pos0 = self.index2d_256(offset, x0, y0);
    //     let mut lut_pos1 = self.index2d_256(offset, x1, y0);

    //     let lx0x = lerp(
    //         CELL_2D_X[lut_pos0 as usize],
    //         CELL_2D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     let ly0x = lerp(
    //         CELL_2D_Y[lut_pos0 as usize],
    //         CELL_2D_Y[lut_pos1 as usize],
    //         xs,
    //     );

    //     lut_pos0 = self.index2d_256(offset, x0, y1);
    //     lut_pos1 = self.index2d_256(offset, x1, y1);

    //     let lx1x = lerp(
    //         CELL_2D_X[lut_pos0 as usize],
    //         CELL_2D_X[lut_pos1 as usize],
    //         xs,
    //     );
    //     let ly1x = lerp(
    //         CELL_2D_Y[lut_pos0 as usize],
    //         CELL_2D_Y[lut_pos1 as usize],
    //         xs,
    //     );

    //     *x += lerp(lx0x, lx1x, ys) * warp_amp;
    //     *y += lerp(ly0x, ly1x, ys) * warp_amp;
    // }
}

#[cfg(test)]
mod tests {
    use glam::vec2;

    use super::{CellularDistanceFunction, FastNoise, NoiseType};

    #[test]
    // Tests that we make an RGB triplet at defaults and it is black.
    fn test_cellular_noise_overflow() {
        let mut noise = FastNoise::seeded(6000);
        noise.set_noise_type(NoiseType::Cellular);
        noise.set_frequency(0.08);
        noise.set_cellular_distance_function(CellularDistanceFunction::Manhattan);
        for y in 0..1024 {
            for x in 0..1024 {
                let frac_x = x as f32 / 1024.0;
                let frac_y = y as f32 / 1024.0;

                let cell_value_f = noise.get_noise(vec2(frac_x, frac_y));
                assert!(cell_value_f != 0.0);
            }
        }
    }
}
