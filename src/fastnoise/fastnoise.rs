// A port of Auburn's FastNoise to Rust.
// I really didn't like the noise libraries I could find, so I ported the one I like.
// Original code: https://github.com/Auburns/FastNoise
// The original is MIT licensed, so this is compatible.

use glam::{IVec3, Vec3A, ivec3, vec3a, vec4};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Deserialize, Serialize};
use std::{f32, fmt::Debug};

use super::{consts::*, enums::*, utils::*};

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
        // let mut rng = rand::rng();
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
        self.fractal_bounding = amp_fractal.recip();
    }

    pub fn set_cellular_distance_indices(&mut self, i1: i32, i2: i32) {
        self.cellular_distance_index.0 = i32::min(i1, i2);
        self.cellular_distance_index.1 = i32::max(i1, i2);

        self.cellular_distance_index.0 = i32::min(
            i32::max(self.cellular_distance_index.0, 0),
            FN_CELLULAR_INDEX_MAX as i32,
        );
        self.cellular_distance_index.1 = i32::min(
            i32::max(self.cellular_distance_index.1, 0),
            FN_CELLULAR_INDEX_MAX as i32,
        );
    }

    pub fn index2d_12(&self, offset: u8, x: i32, y: i32) -> u8 {
        self.perm12[(x & 0xff) as usize + self.perm[(y & 0xff) as usize + offset as usize] as usize]
    }

    #[inline]
    pub fn index3d_12(&self, offset: u8, v: IVec3) -> u8 {
        let z = (v.z as usize & 0xFF) + offset as usize;
        let y = (v.y as usize & 0xFF) + self.perm[z] as usize;
        let x = (v.x as usize & 0xFF) + self.perm[y] as usize;
        self.perm12[x]
    }

    pub fn index2d_256(&self, offset: u8, x: i32, y: i32) -> u8 {
        self.perm[(x as usize & 0xff) + self.perm[(y as usize & 0xff) + offset as usize] as usize]
    }

    #[inline]
    pub fn index3d_256(&self, offset: u8, pos: IVec3) -> u8 {
        let z = (pos.z as usize & 0xFF) + offset as usize;
        let y = (pos.y as usize & 0xFF) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xFF) + self.perm[y] as usize;
        self.perm[x]
    }

    fn val_coord_2d(&self, seed: i32, x: i32, y: i32) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(x);
        n ^= Wrapping(Y_PRIME) * Wrapping(y);
        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    #[inline]
    fn val_coord_3d(&self, seed: i32, pos: IVec3) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(pos.x);
        n ^= Wrapping(Y_PRIME) * Wrapping(pos.y);
        n ^= Wrapping(Z_PRIME) * Wrapping(pos.z);

        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    fn val_coord_2d_fast(&self, offset: u8, x: i32, y: i32) -> f32 {
        VAL_LUT[self.index2d_256(offset, x, y) as usize]
    }

    #[inline]
    fn val_coord_3d_fast(&self, offset: u8, pos: IVec3) -> f32 {
        VAL_LUT[self.index3d_256(offset, pos) as usize]
    }

    fn grad_coord_2d(&self, offset: u8, x: i32, y: i32, xd: f32, yd: f32) -> f32 {
        let lut_pos = self.index2d_12(offset, x, y) as usize;
        xd * GRAD_X[lut_pos] + yd * GRAD_Y[lut_pos]
    }

    #[inline]
    fn grad_coord_3d(&self, offset: u8, a: IVec3, b: Vec3A) -> f32 {
        let lut_pos = self.index3d_12(offset, a) as usize;
        (b * GRAD_A[lut_pos]).element_sum()
    }

    pub fn noise3d<V: Into<Vec3A>>(&self, pos: V) -> f32 {
        let pos = pos.into() * self.frequency;
        match self.noise_type {
            NoiseType::Cellular => match self.cellular_return_type {
                CellularReturnType::CellValue => self.single_cellular3d(pos),
                CellularReturnType::Distance => self.single_cellular3d(pos),
                _ => self.single_cellular_2edge3d(pos),
            },
            NoiseType::Cubic => self.single_cubic3d(0, pos),
            NoiseType::CubicFractal => match self.fractal_type {
                FractalType::FBM => self.single_cubic_fractal_fbm3d(pos),
                FractalType::Billow => self.single_cubic_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(pos),
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
            NoiseType::Value => self.single_value3d(0, pos),
            NoiseType::ValueFractal => match self.fractal_type {
                FractalType::FBM => self.single_value_fractal_fbm3d(pos),
                FractalType::Billow => self.single_value_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(pos),
            },
            NoiseType::WhiteNoise => self.get_white_noise3d(pos),
        }
    }

    #[deprecated(since="0.2.0",note="use noise3d instead")]
    pub fn get_noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.noise3d(vec3a(x, y, z))
    }

    pub fn get_noise(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.noise_type {
            NoiseType::Value => self.single_value(0, x, y),
            NoiseType::ValueFractal => match self.fractal_type {
                FractalType::FBM => self.single_value_fractal_fbm(x, y),
                FractalType::Billow => self.single_value_fractal_billow(x, y),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi(x, y),
            },
            NoiseType::Perlin => self.single_perlin(0, x, y),
            NoiseType::PerlinFractal => match self.fractal_type {
                FractalType::FBM => self.single_perlin_fractal_fbm(x, y),
                FractalType::Billow => self.single_perlin_fractal_billow(x, y),
                FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi(x, y),
            },
            NoiseType::Simplex => self.single_simplex(0, x, y),
            NoiseType::SimplexFractal => match self.fractal_type {
                FractalType::FBM => self.single_simplex_fractal_fbm(x, y),
                FractalType::Billow => self.single_simplex_fractal_billow(x, y),
                FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi(x, y),
            },
            NoiseType::Cellular => match self.cellular_return_type {
                CellularReturnType::CellValue => self.single_cellular(x, y),
                CellularReturnType::Distance => self.single_cellular(x, y),
                _ => self.single_cellular_2edge(x, y),
            },
            NoiseType::WhiteNoise => self.get_white_noise(x, y),
            NoiseType::Cubic => self.single_cubic(0, x, y),
            NoiseType::CubicFractal => match self.fractal_type {
                FractalType::FBM => self.single_cubic_fractal_fbm(x, y),
                FractalType::Billow => self.single_cubic_fractal_billow(x, y),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi(x, y),
            },
        }
    }

    fn get_white_noise3d(&self, pos: Vec3A) -> f32 {
        let c = ivec3(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
        );
        self.val_coord_3d(self.seed as i32, c ^ (c >> 16))
    }

    fn get_white_noise(&self, x: f32, y: f32) -> f32 {
        let xc: i32 = x.to_bits() as i32;
        let yc: i32 = y.to_bits() as i32;

        self.val_coord_2d(self.seed as i32, xc ^ (xc >> 16), yc ^ (yc >> 16))
    }

    // Value noise
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
        let mut sum: f32 = self.single_value3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i: i32 = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_value3d(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - self.single_value3d(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_value3d(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }
        sum
    }

    fn single_value3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor();
        let p1 = (p0 + 1.0).as_ivec3();
        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec(pos - p0),
            Interp::Quintic => interp_quintic_func_vec(pos - p0),
        };

        let p0 = p0.as_ivec3();
        let q0 = vec4(
            self.val_coord_3d_fast(offset, p0),
            self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p0.z)),
            self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p1.z)),
            self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p1.z)),
        );
        let q1 = vec4(
            self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p0.z)),
            self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p0.z)),
            self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p1.z)),
            self.val_coord_3d_fast(offset, p1),
        );

        let qf = q0.lerp(q1, ps.x);
        let yf0 = lerp(qf.x, qf.y, ps.y);
        let yf1 = lerp(qf.z, qf.w, ps.y);

        lerp(yf0, yf1, ps.z)
    }

    fn single_value_fractal_fbm(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum: f32 = self.single_value(self.perm[0], x, y);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_value(self.perm[i as usize], x, y) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_billow(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum: f32 = fast_abs_f(self.single_value(self.perm[0], x, y)) * 2.0 - 1.0;
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            amp *= self.gain;
            sum += (fast_abs_f(self.single_value(self.perm[i as usize], x, y)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_rigid_multi(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum: f32 = 1.0 - fast_abs_f(self.single_value(self.perm[0], x, y));
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += -(1.0 - fast_abs_f(self.single_value(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    fn single_value(&self, offset: u8, x: f32, y: f32) -> f32 {
        let x0 = fast_floor(x);
        let y0 = fast_floor(y);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let xs: f32;
        let ys: f32;
        match self.interp {
            Interp::Linear => {
                xs = x - x0 as f32;
                ys = y - y0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(x - x0 as f32);
                ys = interp_hermite_func(y - y0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(x - x0 as f32);
                ys = interp_quintic_func(y - y0 as f32);
            }
        }

        let xf0 = lerp(
            self.val_coord_2d_fast(offset, x0, y0),
            self.val_coord_2d_fast(offset, x1, y0),
            xs,
        );
        let xf1 = lerp(
            self.val_coord_2d_fast(offset, x0, y1),
            self.val_coord_2d_fast(offset, x1, y1),
            xs,
        );

        lerp(xf0, xf1, ys)
    }

    // Perlin noise
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
        let mut sum: f32 = self.single_perlin3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - self.single_perlin3d(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_perlin3d(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    fn single_perlin3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor();
        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec(pos - p0),
            Interp::Quintic => interp_quintic_func_vec(pos - p0),
        };

        let d0 = pos - p0;
        let d1 = d0 - 1.0;

        let p0 = p0.as_ivec3();
        let p1 = p0 + 1;

        let q0 = vec4(
            self.grad_coord_3d(offset, p0, d0),
            self.grad_coord_3d(offset, ivec3(p0.x, p1.y, p0.z), vec3a(d0.x, d1.y, d0.z)),
            self.grad_coord_3d(offset, ivec3(p0.x, p0.y, p1.z), vec3a(d0.x, d0.y, d1.z)),
            self.grad_coord_3d(offset, ivec3(p0.x, p1.y, p1.z), vec3a(d0.x, d1.y, d1.z)),
        );
        let q1 = vec4(
            self.grad_coord_3d(offset, ivec3(p1.x, p0.y, p0.z), vec3a(d1.x, d0.y, d0.z)),
            self.grad_coord_3d(offset, ivec3(p1.x, p1.y, p0.z), vec3a(d1.x, d1.y, d0.z)),
            self.grad_coord_3d(offset, ivec3(p1.x, p0.y, p1.z), vec3a(d1.x, d0.y, d1.z)),
            self.grad_coord_3d(offset, ivec3(p1.x, p1.y, p1.z), d1),
        );

        let qf = q0.lerp(q1, ps.x);
        let yf0 = lerp(qf.x, qf.y, ps.y);
        let yf1 = lerp(qf.z, qf.w, ps.y);

        lerp(yf0, yf1, ps.z)
    }

    fn single_perlin_fractal_fbm(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = self.single_perlin(self.perm[0], x, y);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_perlin(self.perm[i as usize], x, y) * amp;

            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_billow(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = fast_abs_f(self.single_perlin(self.perm[0], x, y)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += (fast_abs_f(self.single_perlin(self.perm[i as usize], x, y)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_rigid_multi(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_perlin(self.perm[0], x, y));
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += -(1.0 - fast_abs_f(self.single_perlin(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    fn single_perlin(&self, offset: u8, x: f32, y: f32) -> f32 {
        let x0 = fast_floor(x);
        let y0 = fast_floor(y);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let xs: f32;
        let ys: f32;

        match self.interp {
            Interp::Linear => {
                xs = x - x0 as f32;
                ys = y - y0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(x - x0 as f32);
                ys = interp_hermite_func(y - y0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(x - x0 as f32);
                ys = interp_quintic_func(y - y0 as f32);
            }
        }

        let xd0 = x - x0 as f32;
        let yd0 = y - y0 as f32;
        let xd1 = xd0 - 1.0;
        let yd1 = yd0 - 1.0;

        let xf0 = lerp(
            self.grad_coord_2d(offset, x0, y0, xd0, yd0),
            self.grad_coord_2d(offset, x1, y0, xd1, yd0),
            xs,
        );
        let xf1 = lerp(
            self.grad_coord_2d(offset, x0, y1, xd0, yd1),
            self.grad_coord_2d(offset, x1, y1, xd1, yd1),
            xs,
        );

        lerp(xf0, xf1, ys)
    }


    // Simplex noise
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
        let mut sum = self.single_simplex3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += amp * self.single_simplex3d(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0);
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - self.single_simplex3d(self.perm[0], pos).abs();
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_simplex3d(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    fn single_simplex3d(&self, offset: u8, p: Vec3A) -> f32 {

        let mut t = p.element_sum() * F3;
        let q = (p + t).floor();

        t = q.element_sum() * G3;

        let p0 = p - (q - t);

        let (q1, q2) = if p0.x >= p0.y {
            if p0.z <= p0.y {
                (V3A_100, V3A_110)
            } else if p0.z <= p0.x {
                (V3A_100, V3A_101)
            } else {
                (V3A_001, V3A_101)
            }
        } else {
            if p0.z > p0.y {
                (V3A_001, V3A_011)
            } else if p0.z > p0.x {
                (V3A_010, V3A_011)
            } else {
                (V3A_010, V3A_110)
            }
        };

        let p1 = p0 - q1 + G3;
        let p2 = p0 - q2 + 2.0 * G3;
        let p3 = p0 - 1.0 + 3.0 * G3;

        let q = q.as_ivec3();
        let q1 = q1.as_ivec3();
        let q2 = q2.as_ivec3();

        t = 0.6 - (p0 * p0).element_sum();
        let n0 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d(offset, q, p0)
            }
        };

        t = 0.6 - (p1 * p1).element_sum();
        let n1 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d(offset, q + q1, p1)
            }
        };

        t = 0.6 - (p2 * p2).element_sum();
        let n2 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d(offset, q + q2, p2)
            }
        };

        t = 0.6 - (p3 * p3).element_sum();
        let n3 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d(offset, q + 1, p3)
            }
        };

        32.0 * (n0 + n1 + n2 + n3)
    }

    fn single_simplex_fractal_fbm(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = self.single_simplex(self.perm[0], x, y);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_simplex(self.perm[i as usize], x, y) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_billow(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = fast_abs_f(self.single_simplex(self.perm[0], x, y)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += (fast_abs_f(self.single_simplex(self.perm[i as usize], x, y)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_rigid_multi(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_simplex(self.perm[0], x, y));
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += -(1.0 - self.single_simplex(self.perm[i as usize], x, y)) * amp;
            i += 1;
        }

        sum
    }

    fn single_simplex(&self, offset: u8, x: f32, y: f32) -> f32 {
        let mut t: f32 = (x + y) * F2;
        let i = fast_floor(x + t);
        let j = fast_floor(y + t);

        t = (i + j) as f32 * G2;
        let x0 = i as f32 - t;
        let y0 = j as f32 - t;

        let x0 = x - x0;
        let y0 = y - y0;

        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f32 + G2;
        let y1 = y0 - j1 as f32 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let n0;
        let n1;
        let n2;

        t = 0.5 - x0 * x0 - y0 * y0;
        if t < 0. {
            n0 = 0.
        } else {
            t *= t;
            n0 = t * t * self.grad_coord_2d(offset, i, j, x0, y0);
        }

        t = 0.5 - x1 * x1 - y1 * y1;
        if t < 0. {
            n1 = 0.
        } else {
            t *= t;
            n1 = t * t * self.grad_coord_2d(offset, i + i1 as i32, j + j1 as i32, x1, y1);
        }

        t = 0.5 - x2 * x2 - y2 * y2;
        if t < 0. {
            n2 = 0.
        } else {
            t *= t;
            n2 = t * t * self.grad_coord_2d(offset, i + 1, j + 1, x2, y2);
        }

        70.0 * (n0 + n1 + n2)
    }

    // Cubic Noise
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
        let mut sum = self.single_cubic3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_cubic3d(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - self.single_cubic3d(self.perm[0], pos).abs();
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_cubic3d(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    fn single_cubic3d(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor().as_ivec3();
        let p1 = p0 - 1;
        let p2 = p0 + 1;
        let p3 = p0 + 2;
        let p5 = pos - p0.as_vec3a();

        cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p1.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p1.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p0.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p0.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p2.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p2.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p2.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p2.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p3.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p3.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p3.y, p1.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p3.y, p1.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p1.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p1.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p0.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p0.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p2.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p2.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p2.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p2.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p3.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p3.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p3.y, p0.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p3.y, p0.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p1.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p1.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p0.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p0.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p2.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p2.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p2.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p2.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p3.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p3.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p3.y, p2.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p3.y, p2.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p1.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p1.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p1.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p1.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p0.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p0.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p0.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p0.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p2.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p2.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p2.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p2.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, ivec3(p1.x, p3.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p0.x, p3.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p2.x, p3.y, p3.z)),
                    self.val_coord_3d_fast(offset, ivec3(p3.x, p3.y, p3.z)),
                    p5.x,
                ),
                p5.y,
            ),
            p5.z,
        ) * CUBIC_3D_BOUNDING
    }

    fn single_cubic_fractal_fbm(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = self.single_cubic(self.perm[0], x, y);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_cubic(self.perm[i as usize], x, y) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_billow(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = fast_abs_f(self.single_cubic(self.perm[0], x, y)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += (fast_abs_f(self.single_cubic(self.perm[i as usize], x, y)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_rigid_multi(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_cubic(self.perm[0], x, y));
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += -(1.0 - fast_abs_f(self.single_cubic(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    fn single_cubic(&self, offset: u8, x: f32, y: f32) -> f32 {
        let x1 = fast_floor(x);
        let y1 = fast_floor(y);

        let x0 = x1 - 1;
        let y0 = y1 - 1;
        let x2 = x1 + 1;
        let y2 = y1 + 1;
        let x3 = x1 + 2;
        let y3 = y1 + 2;

        let xs = x - x1 as f32;
        let ys = y - y1 as f32;

        cubic_lerp(
            cubic_lerp(
                self.val_coord_2d_fast(offset, x0, y0),
                self.val_coord_2d_fast(offset, x1, y0),
                self.val_coord_2d_fast(offset, x2, y0),
                self.val_coord_2d_fast(offset, x3, y0),
                xs,
            ),
            cubic_lerp(
                self.val_coord_2d_fast(offset, x0, y1),
                self.val_coord_2d_fast(offset, x1, y1),
                self.val_coord_2d_fast(offset, x2, y1),
                self.val_coord_2d_fast(offset, x3, y1),
                xs,
            ),
            cubic_lerp(
                self.val_coord_2d_fast(offset, x0, y2),
                self.val_coord_2d_fast(offset, x1, y2),
                self.val_coord_2d_fast(offset, x2, y2),
                self.val_coord_2d_fast(offset, x3, y2),
                xs,
            ),
            cubic_lerp(
                self.val_coord_2d_fast(offset, x0, y3),
                self.val_coord_2d_fast(offset, x1, y3),
                self.val_coord_2d_fast(offset, x2, y3),
                self.val_coord_2d_fast(offset, x3, y3),
                xs,
            ),
            ys,
        ) * CUBIC_2D_BOUNDING
    }


    // Cellular Noise
    fn single_cellular3d(&self, pos: Vec3A) -> f32 {
        let [xr, yr, zr] = pos.as_ivec3().to_array();

        let mut distance: f32 = f32::MAX;
        let mut c = IVec3::ZERO;
        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = (vec * vec).element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum() + (vec * vec).element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
        }

        match self.cellular_return_type {
            CellularReturnType::CellValue => self.val_coord_3d(self.seed as i32, c),
            CellularReturnType::Distance => distance,
            _ => 0.0,
        }
    }

    fn single_cellular_2edge3d(&self, pos: Vec3A) -> f32 {
        let [xr, yr, zr] = pos.as_ivec3().to_array();

        let mut distance: Vec<f32> = vec![f32::MAX; FN_CELLULAR_INDEX_MAX + 1];

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = (vec * vec).element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
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
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
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
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = self.index3d_256(0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum() + (vec * vec).element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
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

    fn single_cellular(&self, x: f32, y: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);

        let mut distance: f32 = 999999.0;

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = self.index2d_256(0, xi, yi);

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
                        let lut_pos: u8 = self.index2d_256(0, xi, yi);

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
                        let lut_pos: u8 = self.index2d_256(0, xi, yi);

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
                self.val_coord_2d(self.seed as i32, x as i32, y as i32)
            }
            _ => 0.0,
        }
    }

    fn single_cellular_2edge(&self, x: f32, y: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);

        let mut distance: Vec<f32> = vec![999999.0; FN_CELLULAR_INDEX_MAX + 1];

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos = self.index2d_256(0, xi, yi);

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
                        let lut_pos = self.index2d_256(0, xi, yi);

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
                        let lut_pos = self.index2d_256(0, xi, yi);

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

}

#[cfg(test)]
mod tests {
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

                let cell_value_f = noise.get_noise(frac_x, frac_y);
                assert!(cell_value_f != 0.0);
            }
        }
    }
}
