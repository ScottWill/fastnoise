// A port of Auburn's FastNoise to Rust.
// I really didn't like the noise libraries I could find, so I ported the one I like.
// Original code: https://github.com/Auburns/FastNoise
// The original is MIT licensed, so this is compatible.

use glam::{IVec3, Vec3A, ivec3, vec3a};
use rand::Rng;
use rand_pcg::Pcg64;
use rand_seeder::Seeder;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
        self.fractal_bounding = 1.0 / amp_fractal;
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

    pub fn index3d_12(&self, offset: u8, x: i32, y: i32, z: i32) -> u8 {
        let z = (z as usize & 0xff) + offset as usize;
        let y = (y as usize & 0xff) + self.perm[z] as usize;
        let z = (x as usize & 0xff) + self.perm[y] as usize;
        self.perm12[z]
    }

    pub fn index3d_12_vec(&self, offset: u8, v: IVec3) -> u8 {
        let z = (v.z as usize & 0xFF) + offset as usize;
        let y = (v.y as usize & 0xFF) + self.perm[z] as usize;
        let x = (v.x as usize & 0xFF) + self.perm[y] as usize;
        self.perm12[x]
    }

    pub fn index4d_32(&self, offset: u8, x: i32, y: i32, z: i32, w: i32) -> u8 {
        self.perm[(x as usize & 0xff)
            + self.perm[(y as usize & 0xff)
                + self.perm[(z as usize & 0xff)
                    + self.perm[(w as usize & 0xff) + offset as usize] as usize]
                    as usize] as usize]
            & 31
    }

    pub fn index2d_256(&self, offset: u8, x: i32, y: i32) -> u8 {
        self.perm[(x as usize & 0xff) + self.perm[(y as usize & 0xff) + offset as usize] as usize]
    }

    pub fn index3d_256(&self, offset: u8, x: i32, y: i32, z: i32) -> u8 {
        self.perm[(x as usize & 0xff)
            + self.perm
                [(y as usize & 0xff) + self.perm[(z as usize & 0xff) + offset as usize] as usize]
                as usize]
    }

    pub fn index3d_256_vec(&self, offset: u8, pos: IVec3) -> u8 {
        let z = (pos.z as usize & 0xFF) + offset as usize;
        let y = (pos.y as usize & 0xFF) + self.perm[z] as usize;
        let x = (pos.x as usize & 0xFF) + self.perm[y] as usize;
        self.perm[x]
    }

    pub fn index4d_256(&self, offset: u8, x: i32, y: i32, z: i32, w: i32) -> u8 {
        self.perm[(x as usize & 0xff)
            + self.perm[(y as usize & 0xff)
                + self.perm[(z as usize & 0xff)
                    + self.perm[(w as usize & 0xff) + offset as usize] as usize]
                    as usize] as usize]
    }

    fn val_coord_2d(&self, seed: i32, x: i32, y: i32) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(x);
        n ^= Wrapping(Y_PRIME) * Wrapping(y);
        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    fn val_coord_3d(&self, seed: i32, x: i32, y: i32, z: i32) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(x);
        n ^= Wrapping(Y_PRIME) * Wrapping(y);
        n ^= Wrapping(Z_PRIME) * Wrapping(z);

        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    fn val_coord_3d_vec(&self, seed: i32, pos: IVec3) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(pos.x);
        n ^= Wrapping(Y_PRIME) * Wrapping(pos.y);
        n ^= Wrapping(Z_PRIME) * Wrapping(pos.z);

        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    #[allow(dead_code)]
    #[allow(clippy::many_single_char_names)]
    fn val_coord_4d(&self, seed: i32, x: i32, y: i32, z: i32, w: i32) -> f32 {
        use std::num::Wrapping;

        let mut n = Wrapping(seed);
        n ^= Wrapping(X_PRIME) * Wrapping(x);
        n ^= Wrapping(Y_PRIME) * Wrapping(y);
        n ^= Wrapping(Z_PRIME) * Wrapping(z);
        n ^= Wrapping(W_PRIME) * Wrapping(w);

        (n * n * n * Wrapping(60493i32)).0 as f32 / 2147483648.0
    }

    fn val_coord_2d_fast(&self, offset: u8, x: i32, y: i32) -> f32 {
        VAL_LUT[self.index2d_256(offset, x, y) as usize]
    }
    fn val_coord_3d_fast(&self, offset: u8, x: i32, y: i32, z: i32) -> f32 {
        VAL_LUT[self.index3d_256(offset, x, y, z) as usize]
    }
    fn val_coord_3d_fast_vec(&self, offset: u8, pos: IVec3) -> f32 {
        VAL_LUT[self.index3d_256_vec(offset, pos) as usize]
    }

    fn grad_coord_2d(&self, offset: u8, x: i32, y: i32, xd: f32, yd: f32) -> f32 {
        let lut_pos = self.index2d_12(offset, x, y) as usize;
        xd * GRAD_X[lut_pos] + yd * GRAD_Y[lut_pos]
    }

    fn grad_coord_3d_vec(&self, offset: u8, a: IVec3, b: Vec3A) -> f32 {
        let lut_pos = self.index3d_12_vec(offset, a) as usize;
        (b * GRAD_A[lut_pos]).element_sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn grad_coord_3d(&self, offset: u8, x: i32, y: i32, z: i32, xd: f32, yd: f32, zd: f32) -> f32 {
        let lut_pos = self.index3d_12(offset, x, y, z) as usize;
        xd * GRAD_X[lut_pos] + yd * GRAD_Y[lut_pos] + zd * GRAD_Z[lut_pos]
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn grad_coord_4d(
        &self,
        offset: u8,
        x: i32,
        y: i32,
        z: i32,
        w: i32,
        xd: f32,
        yd: f32,
        zd: f32,
        wd: f32,
    ) -> f32 {
        let lut_pos = self.index4d_32(offset, x, y, z, w) as usize;
        xd * GRAD_4D[lut_pos]
            + yd * GRAD_4D[lut_pos + 1]
            + zd * GRAD_4D[lut_pos + 2]
            + wd * GRAD_4D[lut_pos + 3]
    }

    pub fn get_noise3d_vec(&self, mut pos: Vec3A) -> f32 {
        pos *= self.frequency;

        match self.noise_type {
            NoiseType::Cubic => self.single_cubic3d_vec(0, pos),
            NoiseType::CubicFractal => match self.fractal_type {
                FractalType::FBM => self.single_cubic_fractal_fbm3d_vec(pos),
                FractalType::Billow => self.single_cubic_fractal_billow3d_vec(pos),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d_vec(pos),
            },
            NoiseType::Perlin => self.single_perlin3d_vec(0, pos),
            NoiseType::PerlinFractal => match self.fractal_type {
                FractalType::FBM => self.single_perlin_fractal_fbm3d_vec(pos),
                FractalType::Billow => self.single_perlin_fractal_billow3d_vec(pos),
                FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d_vec(pos),
            },
            NoiseType::Simplex => self.single_simplex3d_vec(0, pos),
            NoiseType::SimplexFractal => match self.fractal_type {
                FractalType::FBM => self.single_simplex_fractal_fbm3d_vec(pos),
                FractalType::Billow => self.single_simplex_fractal_billow3d_vec(pos),
                FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi3d_vec(pos),
            },
            NoiseType::Value => self.single_value3d_vec(0, pos),
            NoiseType::ValueFractal => match self.fractal_type {
                FractalType::FBM => self.single_value_fractal_fbm3d_vec(pos),
                FractalType::Billow => self.single_value_fractal_billow3d_vec(pos),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d_vec(pos),
            },
            NoiseType::WhiteNoise => self.get_white_noise3d_vec(pos),
            _ => todo!(),
        }
    }

    pub fn get_noise3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.noise_type {
            NoiseType::Value => self.single_value3d(0, x, y, z),
            NoiseType::ValueFractal => match self.fractal_type {
                FractalType::FBM => self.single_value_fractal_fbm3d(x, y, z),
                FractalType::Billow => self.single_value_fractal_billow3d(x, y, z),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(x, y, z),
            },
            NoiseType::Perlin => self.single_perlin3d(0, x, y, z),
            NoiseType::PerlinFractal => match self.fractal_type {
                FractalType::FBM => self.single_perlin_fractal_fbm3d(x, y, z),
                FractalType::Billow => self.single_perlin_fractal_billow3d(x, y, z),
                FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d(x, y, z),
            },
            NoiseType::Simplex => self.single_simplex3d(0, x, y, z),
            NoiseType::SimplexFractal => match self.fractal_type {
                FractalType::FBM => self.single_simplex_fractal_fbm3d(x, y, z),
                FractalType::Billow => self.single_simplex_fractal_billow3d(x, y, z),
                FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi3d(x, y, z),
            },
            NoiseType::Cellular => match self.cellular_return_type {
                CellularReturnType::CellValue => self.single_cellular3d(x, y, z),
                CellularReturnType::Distance => self.single_cellular3d(x, y, z),
                _ => self.single_cellular_2edge3d(x, y, z),
            },
            NoiseType::WhiteNoise => self.get_white_noise3d(x, y, z),
            NoiseType::Cubic => self.single_cubic3d(0, x, y, z),
            NoiseType::CubicFractal => match self.fractal_type {
                FractalType::FBM => self.single_cubic_fractal_fbm3d(x, y, z),
                FractalType::Billow => self.single_cubic_fractal_billow3d(x, y, z),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(x, y, z),
            },
        }
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

    #[allow(dead_code)]
    fn get_white_noise4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 {
        let xc: i32 = x.to_bits() as i32;
        let yc: i32 = y.to_bits() as i32;
        let zc: i32 = z.to_bits() as i32;
        let wc: i32 = w.to_bits() as i32;

        self.val_coord_4d(
            self.seed as i32,
            xc ^ (xc as i32 >> 16),
            yc ^ (yc >> 16),
            zc ^ (zc >> 16),
            wc ^ (wc >> 16),
        )
    }

    fn get_white_noise3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xc: i32 = x.to_bits() as i32;
        let yc: i32 = y.to_bits() as i32;
        let zc: i32 = z.to_bits() as i32;

        self.val_coord_3d(
            self.seed as i32,
            xc ^ (xc >> 16),
            yc ^ (yc >> 16),
            zc ^ (zc >> 16),
        )
    }

    fn get_white_noise3d_vec(&self, pos: Vec3A) -> f32 {
        let c = ivec3(
            pos.x.to_bits() as i32,
            pos.y.to_bits() as i32,
            pos.z.to_bits() as i32,
        );
        self.val_coord_3d_vec(self.seed as i32, c ^ (c >> 16))
    }

    fn get_white_noise(&self, x: f32, y: f32) -> f32 {
        let xc: i32 = x.to_bits() as i32;
        let yc: i32 = y.to_bits() as i32;

        self.val_coord_2d(self.seed as i32, xc ^ (xc >> 16), yc ^ (yc >> 16))
    }

    #[allow(dead_code)]
    fn get_white_noise_int4d(&self, x: i32, y: i32, z: i32, w: i32) -> f32 {
        self.val_coord_4d(self.seed as i32, x, y, z, w)
    }

    #[allow(dead_code)]
    fn get_white_noise_int3d(&self, x: i32, y: i32, z: i32) -> f32 {
        self.val_coord_3d(self.seed as i32, x, y, z)
    }

    #[allow(dead_code)]
    fn get_white_noise_int(&self, x: i32, y: i32) -> f32 {
        self.val_coord_2d(self.seed as i32, x, y)
    }

    #[allow(dead_code)]
    // Value noise
    fn get_value_fractal3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_value_fractal_fbm3d(x, y, z),
            FractalType::Billow => self.single_value_fractal_billow3d(x, y, z),
            FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(x, y, z),
        }
    }

    #[allow(dead_code)]
    fn get_value_fractal(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_value_fractal_fbm(x, y),
            FractalType::Billow => self.single_value_fractal_billow(x, y),
            FractalType::RigidMulti => self.single_value_fractal_rigid_multi(x, y),
        }
    }

    fn single_value_fractal_fbm3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = self.single_value3d(self.perm[0], x, y, z);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_value3d(self.perm[i as usize], x, y, z) * amp;

            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_fbm3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_value3d_vec(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_value3d_vec(self.perm[i as usize], pos) * amp;

            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_billow3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = fast_abs_f(self.single_value3d(self.perm[0], x, y, z)) * 2.0 - 1.0;
        let mut amp: f32 = 1.0;
        let mut i: i32 = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum +=
                (fast_abs_f(self.single_value3d(self.perm[i as usize], x, y, z)) * 2.0 - 1.0) * amp;

            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_billow3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_value3d_vec(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i: i32 = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_value3d_vec(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_rigid_multi3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = 1.0 - fast_abs_f(self.single_value3d(self.perm[0], x, y, z));
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_value3d(self.perm[i as usize], x, y, z))) * amp;

            i += 1;
        }
        sum
    }

    fn single_value_fractal_rigid_multi3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - self.single_value3d_vec(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - self.single_value3d_vec(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }
        sum
    }

    #[allow(dead_code)]
    fn get_value3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.single_value3d(
            0,
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
        )
    }

    fn single_value3d(&self, offset: u8, x: f32, y: f32, z: f32) -> f32 {
        let x0 = fast_floor(x);
        let y0 = fast_floor(y);
        let z0 = fast_floor(z);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let xs: f32;
        let ys: f32;
        let zs: f32;
        match self.interp {
            Interp::Linear => {
                xs = x - x0 as f32;
                ys = y - y0 as f32;
                zs = z - z0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(x - x0 as f32);
                ys = interp_hermite_func(y - y0 as f32);
                zs = interp_hermite_func(z - z0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(x - x0 as f32);
                ys = interp_quintic_func(y - y0 as f32);
                zs = interp_quintic_func(z - z0 as f32);
            }
        }

        let xf00: f32 = lerp(
            self.val_coord_3d_fast(offset, x0, y0, z0),
            self.val_coord_3d_fast(offset, x1, y0, z0),
            xs,
        );
        let xf10: f32 = lerp(
            self.val_coord_3d_fast(offset, x0, y1, z0),
            self.val_coord_3d_fast(offset, x1, y1, z0),
            xs,
        );
        let xf01: f32 = lerp(
            self.val_coord_3d_fast(offset, x0, y0, z1),
            self.val_coord_3d_fast(offset, x1, y0, z1),
            xs,
        );
        let xf11: f32 = lerp(
            self.val_coord_3d_fast(offset, x0, y1, z1),
            self.val_coord_3d_fast(offset, x1, y1, z1),
            xs,
        );

        let yf0: f32 = lerp(xf00, xf10, ys);
        let yf1: f32 = lerp(xf01, xf11, ys);

        lerp(yf0, yf1, zs)
    }

    fn single_value3d_vec(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor();
        let p1 = (p0 + 1.0).as_ivec3();
        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec(pos - p0),
            Interp::Quintic => interp_quintic_func_vec(pos - p0),
        };

        let p0 = p0.as_ivec3();
        let xf00: f32 = lerp(
            self.val_coord_3d_fast_vec(offset, p0),
            self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p0.z)),
            ps.x,
        );
        let xf10: f32 = lerp(
            self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p0.z)),
            self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p1.y, p0.z)),
            ps.x,
        );
        let xf01: f32 = lerp(
            self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p0.y, p1.z)),
            self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p1.z)),
            ps.x,
        );
        let xf11: f32 = lerp(
            self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p1.z)),
            self.val_coord_3d_fast_vec(offset, p1),
            ps.x,
        );

        let yf0: f32 = lerp(xf00, xf10, ps.y);
        let yf1: f32 = lerp(xf01, xf11, ps.y);

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
            sum -= (1.0 - fast_abs_f(self.single_value(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_value(&self, x: f32, y: f32) -> f32 {
        self.single_value(0, x * self.frequency, y * self.frequency)
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

    #[allow(dead_code)]
    fn get_perlin_fractal3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_perlin_fractal_fbm3d(x, y, z),
            FractalType::Billow => self.single_perlin_fractal_billow3d(x, y, z),
            FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d(x, y, z),
        }
    }

    fn single_perlin_fractal_fbm3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = self.single_perlin3d(self.perm[0], x, y, z);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_perlin3d(self.perm[i as usize], x, y, z) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_fbm3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_perlin3d_vec(self.perm[0], pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d_vec(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_billow3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = fast_abs_f(self.single_perlin3d(self.perm[0], x, y, z)) * 2.0 - 1.0;
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += (fast_abs_f(self.single_perlin3d(self.perm[i as usize], x, y, z)) * 2.0 - 1.0)
                * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_billow3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_perlin3d_vec(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d_vec(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_perlin_fractal_rigid_multi3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum: f32 = 1.0 - fast_abs_f(self.single_perlin3d(self.perm[0], x, y, z));
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_perlin3d(self.perm[i as usize], x, y, z))) * amp;

            i += 1;
        }

        sum
    }

    fn single_perlin_fractal_rigid_multi3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - self.single_perlin3d_vec(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - self.single_perlin3d_vec(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_perlin3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.single_perlin3d(
            0,
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
        )
    }

    fn single_perlin3d(&self, offset: u8, x: f32, y: f32, z: f32) -> f32 {
        let x0 = fast_floor(x);
        let y0 = fast_floor(y);
        let z0 = fast_floor(z);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let xs: f32;
        let ys: f32;
        let zs: f32;

        match self.interp {
            Interp::Linear => {
                xs = x - x0 as f32;
                ys = y - y0 as f32;
                zs = z - z0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(x - x0 as f32);
                ys = interp_hermite_func(y - y0 as f32);
                zs = interp_hermite_func(z - z0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(x - x0 as f32);
                ys = interp_quintic_func(y - y0 as f32);
                zs = interp_quintic_func(z - z0 as f32);
            }
        }

        let xd0 = x - x0 as f32;
        let yd0 = y - y0 as f32;
        let zd0 = z - z0 as f32;
        let xd1 = xd0 - 1.0;
        let yd1 = yd0 - 1.0;
        let zd1 = zd0 - 1.0;

        let xf00 = lerp(
            self.grad_coord_3d(offset, x0, y0, z0, xd0, yd0, zd0),
            self.grad_coord_3d(offset, x1, y0, z0, xd1, yd0, zd0),
            xs,
        );
        let xf10 = lerp(
            self.grad_coord_3d(offset, x0, y1, z0, xd0, yd1, zd0),
            self.grad_coord_3d(offset, x1, y1, z0, xd1, yd1, zd0),
            xs,
        );
        let xf01 = lerp(
            self.grad_coord_3d(offset, x0, y0, z1, xd0, yd0, zd1),
            self.grad_coord_3d(offset, x1, y0, z1, xd1, yd0, zd1),
            xs,
        );
        let xf11 = lerp(
            self.grad_coord_3d(offset, x0, y1, z1, xd0, yd1, zd1),
            self.grad_coord_3d(offset, x1, y1, z1, xd1, yd1, zd1),
            xs,
        );

        let yf0 = lerp(xf00, xf10, ys);
        let yf1 = lerp(xf01, xf11, ys);

        lerp(yf0, yf1, zs)
    }

    fn single_perlin3d_vec(&self, offset: u8, pos: Vec3A) -> f32 {
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

        let xf00 = lerp(
            self.grad_coord_3d_vec(offset, p0, d0),
            self.grad_coord_3d_vec(offset, ivec3(p1.x, p0.y, p0.z), vec3a(d1.x, d0.y, d0.z)),
            ps.x,
        );
        let xf10 = lerp(
            self.grad_coord_3d_vec(offset, ivec3(p0.x, p1.y, p0.z), vec3a(d0.x, d1.y, d0.z)),
            self.grad_coord_3d_vec(offset, ivec3(p1.x, p1.y, p0.z), vec3a(d1.x, d1.y, d0.z)),
            ps.x,
        );
        let xf01 = lerp(
            self.grad_coord_3d_vec(offset, ivec3(p0.x, p0.y, p1.z), vec3a(d0.x, d0.y, d1.z)),
            self.grad_coord_3d_vec(offset, ivec3(p1.x, p0.y, p1.z), vec3a(d1.x, d0.y, d1.z)),
            ps.x,
        );
        let xf11 = lerp(
            self.grad_coord_3d_vec(offset, ivec3(p0.x, p1.y, p1.z), vec3a(d0.x, d1.y, d1.z)),
            self.grad_coord_3d_vec(offset, ivec3(p1.x, p1.y, p1.z), d1),
            ps.x,
        );

        let yf0 = lerp(xf00, xf10, ps.y);
        let yf1 = lerp(xf01, xf11, ps.y);

        lerp(yf0, yf1, ps.z)
    }

    #[allow(dead_code)]
    fn get_perlin_fractal(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_perlin_fractal_fbm(x, y),
            FractalType::Billow => self.single_perlin_fractal_billow(x, y),
            FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi(x, y),
        }
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
            sum -= (1.0 - fast_abs_f(self.single_perlin(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_perlin(&self, x: f32, y: f32) -> f32 {
        self.single_perlin(0, x * self.frequency, y * self.frequency)
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

    #[allow(dead_code)]
    // Simplex noise
    fn get_simplex_fractal3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_simplex_fractal_fbm3d(x, y, z),
            FractalType::Billow => self.single_simplex_fractal_billow3d(x, y, z),
            FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi3d(x, y, z),
        }
    }

    fn single_simplex_fractal_fbm3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = self.single_simplex3d(self.perm[0], x, y, z);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_simplex3d(self.perm[i as usize], x, y, z) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_fbm3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_simplex3d_vec(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_simplex3d_vec(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_billow3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = fast_abs_f(self.single_simplex3d(self.perm[0], x, y, z)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += (fast_abs_f(self.single_simplex3d(self.perm[i as usize], x, y, z)) * 2.0 - 1.0)
                * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_billow3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_simplex3d_vec(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += amp * self.single_simplex3d_vec(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0);
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_simplex_fractal_rigid_multi3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_simplex3d(self.perm[0], x, y, z));
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_simplex3d(self.perm[i as usize], x, y, z))) * amp;
            i += 1;
        }

        sum
    }

    fn single_simplex_fractal_rigid_multi3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - self.single_simplex3d_vec(self.perm[0], pos).abs();
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - self.single_simplex3d_vec(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_simplex3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.single_simplex3d(
            0,
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
        )
    }

    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::collapsible_if)]
    #[allow(clippy::suspicious_else_formatting)]
    fn single_simplex3d_vec(&self, offset: u8, p: Vec3A) -> f32 {

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
            if p0.y < p0.z {
                (V3A_001, V3A_011)
            } else if p0.x < p0.z {
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
                t * t * self.grad_coord_3d_vec(offset, q, p0)
            }
        };

        t = 0.6 - (p1 * p1).element_sum();
        let n1 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d_vec(offset, q + q1, p1)
            }
        };

        t = 0.6 - (p2 * p2).element_sum();
        let n2 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d_vec(offset, q + q2, p2)
            }
        };

        t = 0.6 - (p3 * p3).element_sum();
        let n3 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * self.grad_coord_3d_vec(offset, q + 1, p3)
            }
        };

        32.0 * (n0 + n1 + n2 + n3)
    }

    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::collapsible_if)]
    #[allow(clippy::suspicious_else_formatting)]
    fn single_simplex3d(&self, offset: u8, x: f32, y: f32, z: f32) -> f32 {
        let mut t: f32 = (x + y + z) * F3;
        let i = fast_floor(x + t);
        let j = fast_floor(y + t);
        let k = fast_floor(z + t);

        t = (i + j + k) as f32 * G3;
        let x0 = i as f32 - t;
        let y0 = j as f32 - t;
        let z0 = k as f32 - t;

        let x0 = x - x0;
        let y0 = y - y0;
        let z0 = z - z0;

        let i1: f32;
        let j1: f32;
        let k1: f32;
        let i2: f32;
        let j2: f32;
        let k2: f32;

        if x0 >= y0 {
            if y0 >= z0 {
                i1 = 1.;
                j1 = 0.;
                k1 = 0.;
                i2 = 1.;
                j2 = 1.;
                k2 = 0.;
            } else if x0 >= z0 {
                i1 = 1.;
                j1 = 0.;
                k1 = 0.;
                i2 = 1.;
                j2 = 0.;
                k2 = 1.;
            } else
            // x0 < z0
            {
                i1 = 0.;
                j1 = 0.;
                k1 = 1.;
                i2 = 1.;
                j2 = 0.;
                k2 = 1.;
            }
        } else
        // x0 < y0
        {
            if y0 < z0 {
                i1 = 0.;
                j1 = 0.;
                k1 = 1.;
                i2 = 0.;
                j2 = 1.;
                k2 = 1.;
            } else if x0 < z0 {
                i1 = 0.;
                j1 = 1.;
                k1 = 0.;
                i2 = 0.;
                j2 = 1.;
                k2 = 1.;
            } else
            // x0 >= z0
            {
                i1 = 0.;
                j1 = 1.;
                k1 = 0.;
                i2 = 1.;
                j2 = 1.;
                k2 = 0.;
            }
        }

        let x1 = x0 - i1 + G3;
        let y1 = y0 - j1 + G3;
        let z1 = z0 - k1 + G3;
        let x2 = x0 - i2 + 2.0 * G3;
        let y2 = y0 - j2 + 2.0 * G3;
        let z2 = z0 - k2 + 2.0 * G3;
        let x3 = x0 - 1. + 3.0 * G3;
        let y3 = y0 - 1. + 3.0 * G3;
        let z3 = z0 - 1. + 3.0 * G3;

        let n0;
        let n1;
        let n2;
        let n3;

        t = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t < 0. {
            n0 = 0.;
        } else {
            t *= t;
            n0 = t * t * self.grad_coord_3d(offset, i, j, k, x0, y0, z0);
        }

        t = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t < 0. {
            n1 = 0.
        } else {
            t *= t;
            n1 = t
                * t
                * self.grad_coord_3d(
                    offset,
                    i + i1 as i32,
                    j + j1 as i32,
                    k + k1 as i32,
                    x1,
                    y1,
                    z1,
                );
        }

        t = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t < 0. {
            n2 = 0.
        } else {
            t *= t;
            n2 = t
                * t
                * self.grad_coord_3d(
                    offset,
                    i + i2 as i32,
                    j + j2 as i32,
                    k + k2 as i32,
                    x2,
                    y2,
                    z2,
                );
        }

        t = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t < 0. {
            n3 = 0.
        } else {
            t *= t;
            n3 = t * t * self.grad_coord_3d(offset, i + 1, j + 1, k + 1, x3, y3, z3);
        }

        32.0 * (n0 + n1 + n2 + n3)
    }

    #[allow(dead_code)]
    fn get_simplex_fractal(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_simplex_fractal_fbm(x, y),
            FractalType::Billow => self.single_simplex_fractal_billow(x, y),
            FractalType::RigidMulti => self.single_simplex_fractal_rigid_multi(x, y),
        }
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
            sum -= (1.0 - self.single_simplex(self.perm[i as usize], x, y)) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn single_simplex_fractal_blend(&self, mut x: f32, mut y: f32) -> f32 {
        let mut sum = self.single_simplex(self.perm[0], x, y);
        let mut amp = 1.0;
        let mut i = 1;

        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_simplex(self.perm[i as usize], x, y) * amp + 1.0;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    #[allow(dead_code)]
    fn get_simplex(&self, x: f32, y: f32) -> f32 {
        self.single_simplex(0, x * self.frequency, y * self.frequency)
    }

    #[allow(clippy::many_single_char_names)]
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

    #[allow(dead_code)]
    fn get_simplex_4d(&self, x: f32, y: f32, z: f32, w: f32) -> f32 {
        self.single_simplex4d(
            0,
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
            w * self.frequency,
        )
    }

    #[allow(dead_code)]
    fn greater_1_0(&self, n: i32, greater_than: i32) -> i32 {
        if n >= greater_than {
            1
        } else {
            0
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::many_single_char_names)]
    fn single_simplex4d(&self, offset: u8, x: f32, y: f32, z: f32, w: f32) -> f32 {
        let n0: f32;
        let n1: f32;
        let n2: f32;
        let n3: f32;
        let n4: f32;

        let mut t = (x + y + z + w) * F4;
        let i = fast_floor(x + t) as f32;
        let j = fast_floor(y + t) as f32;
        let k = fast_floor(z + t) as f32;
        let l = fast_floor(w + t) as f32;
        t = (i + j + k + l) * G4;
        let x0 = i - t;
        let y0 = j - t;
        let z0 = k - t;
        let w0 = l - t;
        let x0 = x - x0;
        let y0 = y - y0;
        let z0 = z - z0;
        let w0 = w - w0;

        let mut rankx = 0;
        let mut ranky = 0;
        let mut rankz = 0;
        let mut rankw = 0;

        if x0 > y0 {
            rankx += 1;
        } else {
            ranky += 1;
        }
        if x0 > z0 {
            rankx += 1;
        } else {
            rankz += 1
        };
        if x0 > w0 {
            rankx += 1;
        } else {
            rankw += 1
        };
        if y0 > z0 {
            ranky += 1;
        } else {
            rankz += 1
        };
        if y0 > w0 {
            ranky += 1;
        } else {
            rankw += 1
        };
        if z0 > w0 {
            rankz += 1;
        } else {
            rankw += 1
        };

        let i1 = self.greater_1_0(rankx, 3);
        let j1 = self.greater_1_0(ranky, 3);
        let k1 = self.greater_1_0(rankz, 3);
        let l1 = self.greater_1_0(rankw, 3);

        let i2 = self.greater_1_0(rankx, 2);
        let j2 = self.greater_1_0(ranky, 2);
        let k2 = self.greater_1_0(rankz, 2);
        let l2 = self.greater_1_0(rankw, 2);

        let i3 = self.greater_1_0(rankx, 1);
        let j3 = self.greater_1_0(ranky, 1);
        let k3 = self.greater_1_0(rankz, 1);
        let l3 = self.greater_1_0(rankw, 1);

        let x1 = x0 - i1 as f32 + G4;
        let y1 = y0 - j1 as f32 + G4;
        let z1 = z0 - k1 as f32 + G4;
        let w1 = w0 - l1 as f32 + G4;
        let x2 = x0 - i2 as f32 + 2.0 * G4;
        let y2 = y0 - j2 as f32 + 2.0 * G4;
        let z2 = z0 - k2 as f32 + 2.0 * G4;
        let w2 = w0 - l2 as f32 + 2.0 * G4;
        let x3 = x0 - i3 as f32 + 3.0 * G4;
        let y3 = y0 - j3 as f32 + 3.0 * G4;
        let z3 = z0 - k3 as f32 + 3.0 * G4;
        let w3 = w0 - l3 as f32 + 3.0 * G4;
        let x4 = x0 - 1.0 + 4.0 * G4;
        let y4 = y0 - 1.0 + 4.0 * G4;
        let z4 = z0 - 1.0 + 4.0 * G4;
        let w4 = w0 - 1.0 + 4.0 * G4;

        t = 0.6 - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
        if t < 0.0 {
            n0 = 0.;
        } else {
            t *= t;
            n0 = t
                * t
                * self.grad_coord_4d(
                    offset, i as i32, j as i32, k as i32, l as i32, x0, y0, z0, w0,
                );
        }
        t = 0.6 - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
        if t < 0.0 {
            n1 = 0.;
        } else {
            t *= t;
            n1 = t
                * t
                * self.grad_coord_4d(
                    offset,
                    i as i32 + i1,
                    j as i32 + j1,
                    k as i32 + k1,
                    l as i32 + l1,
                    x1,
                    y1,
                    z1,
                    w1,
                );
        }
        t = 0.6 - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
        if t < 0.0 {
            n2 = 0.;
        } else {
            t *= t;
            n2 = t
                * t
                * self.grad_coord_4d(
                    offset,
                    i as i32 + i2,
                    j as i32 + j2,
                    k as i32 + k2,
                    l as i32 + l2,
                    x2,
                    y2,
                    z2,
                    w2,
                );
        }
        t = 0.6 - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
        if t < 0.0 {
            n3 = 0.;
        } else {
            t *= t;
            n3 = t
                * t
                * self.grad_coord_4d(
                    offset,
                    i as i32 + i3,
                    j as i32 + j3,
                    k as i32 + k3,
                    l as i32 + l3,
                    x3,
                    y3,
                    z3,
                    w3,
                );
        }
        t = 0.6 - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
        if t < 0.0 {
            n4 = 0.;
        } else {
            t *= t;
            n4 = t
                * t
                * self.grad_coord_4d(
                    offset,
                    i as i32 + 1,
                    j as i32 + 1,
                    k as i32 + 1,
                    l as i32 + 1,
                    x4,
                    y4,
                    z4,
                    w4,
                );
        }

        27.0 * (n0 + n1 + n2 + n3 + n4) as f32
    }

    #[allow(dead_code)]
    // Cubic Noise
    fn get_cubic_fractal3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_cubic_fractal_fbm3d(x, y, z),
            FractalType::Billow => self.single_cubic_fractal_billow3d(x, y, z),
            FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(x, y, z),
        }
    }

    fn single_cubic_fractal_fbm3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = self.single_cubic3d(self.perm[0], x, y, z);
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum += self.single_cubic3d(self.perm[i as usize], x, y, z) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_fbm3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_cubic3d_vec(self.perm[0], pos);
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_cubic3d_vec(self.perm[i as usize], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_billow3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = fast_abs_f(self.single_cubic3d(self.perm[0], x, y, z)) * 2.0 - 1.0;
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum +=
                (fast_abs_f(self.single_cubic3d(self.perm[i as usize], x, y, z)) * 2.0 - 1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_billow3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_cubic3d_vec(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_cubic3d_vec(self.perm[i as usize], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn single_cubic_fractal_rigid_multi3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut sum = 1.0 - fast_abs_f(self.single_cubic3d(self.perm[0], x, y, z));
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;

            amp *= self.gain;
            sum -= (1.0 - fast_abs_f(self.single_cubic3d(self.perm[i as usize], x, y, z))) * amp;
            i += 1;
        }

        sum
    }

    fn single_cubic_fractal_rigid_multi3d_vec(&self, mut pos: Vec3A) -> f32 {
        let mut sum = 1.0 - self.single_cubic3d_vec(self.perm[0], pos).abs();
        let mut amp = 1.0;
        let mut i = 1;
        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - self.single_cubic3d_vec(self.perm[i as usize], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_cubic3d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.single_cubic3d(
            0,
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
        )
    }

    fn single_cubic3d(&self, offset: u8, x: f32, y: f32, z: f32) -> f32 {
        let x1 = fast_floor(x);
        let y1 = fast_floor(y);
        let z1 = fast_floor(z);

        let x0 = x1 - 1;
        let y0 = y1 - 1;
        let z0 = z1 - 1;
        let x2 = x1 + 1;
        let y2 = y1 + 1;
        let z2 = z1 + 1;
        let x3 = x1 + 2;
        let y3 = y1 + 2;
        let z3 = z1 + 2;

        let xs = x - x1 as f32;
        let ys = y - y1 as f32;
        let zs = z - z1 as f32;

        cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y0, z0),
                    self.val_coord_3d_fast(offset, x1, y0, z0),
                    self.val_coord_3d_fast(offset, x2, y0, z0),
                    self.val_coord_3d_fast(offset, x3, y0, z0),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y1, z0),
                    self.val_coord_3d_fast(offset, x1, y1, z0),
                    self.val_coord_3d_fast(offset, x2, y1, z0),
                    self.val_coord_3d_fast(offset, x3, y1, z0),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y2, z0),
                    self.val_coord_3d_fast(offset, x1, y2, z0),
                    self.val_coord_3d_fast(offset, x2, y2, z0),
                    self.val_coord_3d_fast(offset, x3, y2, z0),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y3, z0),
                    self.val_coord_3d_fast(offset, x1, y3, z0),
                    self.val_coord_3d_fast(offset, x2, y3, z0),
                    self.val_coord_3d_fast(offset, x3, y3, z0),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y0, z1),
                    self.val_coord_3d_fast(offset, x1, y0, z1),
                    self.val_coord_3d_fast(offset, x2, y0, z1),
                    self.val_coord_3d_fast(offset, x3, y0, z1),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y1, z1),
                    self.val_coord_3d_fast(offset, x1, y1, z1),
                    self.val_coord_3d_fast(offset, x2, y1, z1),
                    self.val_coord_3d_fast(offset, x3, y1, z1),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y2, z1),
                    self.val_coord_3d_fast(offset, x1, y2, z1),
                    self.val_coord_3d_fast(offset, x2, y2, z1),
                    self.val_coord_3d_fast(offset, x3, y2, z1),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y3, z1),
                    self.val_coord_3d_fast(offset, x1, y3, z1),
                    self.val_coord_3d_fast(offset, x2, y3, z1),
                    self.val_coord_3d_fast(offset, x3, y3, z1),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y0, z2),
                    self.val_coord_3d_fast(offset, x1, y0, z2),
                    self.val_coord_3d_fast(offset, x2, y0, z2),
                    self.val_coord_3d_fast(offset, x3, y0, z2),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y1, z2),
                    self.val_coord_3d_fast(offset, x1, y1, z2),
                    self.val_coord_3d_fast(offset, x2, y1, z2),
                    self.val_coord_3d_fast(offset, x3, y1, z2),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y2, z2),
                    self.val_coord_3d_fast(offset, x1, y2, z2),
                    self.val_coord_3d_fast(offset, x2, y2, z2),
                    self.val_coord_3d_fast(offset, x3, y2, z2),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y3, z2),
                    self.val_coord_3d_fast(offset, x1, y3, z2),
                    self.val_coord_3d_fast(offset, x2, y3, z2),
                    self.val_coord_3d_fast(offset, x3, y3, z2),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y0, z3),
                    self.val_coord_3d_fast(offset, x1, y0, z3),
                    self.val_coord_3d_fast(offset, x2, y0, z3),
                    self.val_coord_3d_fast(offset, x3, y0, z3),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y1, z3),
                    self.val_coord_3d_fast(offset, x1, y1, z3),
                    self.val_coord_3d_fast(offset, x2, y1, z3),
                    self.val_coord_3d_fast(offset, x3, y1, z3),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y2, z3),
                    self.val_coord_3d_fast(offset, x1, y2, z3),
                    self.val_coord_3d_fast(offset, x2, y2, z3),
                    self.val_coord_3d_fast(offset, x3, y2, z3),
                    xs,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast(offset, x0, y3, z3),
                    self.val_coord_3d_fast(offset, x1, y3, z3),
                    self.val_coord_3d_fast(offset, x2, y3, z3),
                    self.val_coord_3d_fast(offset, x3, y3, z3),
                    xs,
                ),
                ys,
            ),
            zs,
        ) * CUBIC_3D_BOUNDING
    }

    fn single_cubic3d_vec(&self, offset: u8, pos: Vec3A) -> f32 {
        let p0 = pos.floor().as_ivec3();
        let p1 = p0 - 1;
        let p2 = p0 + 1;
        let p3 = p0 + 2;
        let p5 = pos - p0.as_vec3a();

        cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p1.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p1.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p1.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p0.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p0.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p0.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p2.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p2.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p2.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p2.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p3.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p3.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p3.y, p1.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p3.y, p1.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p1.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p1.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p1.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p0.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p0.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p0.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p2.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p2.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p2.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p2.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p3.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p3.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p3.y, p0.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p3.y, p0.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p1.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p1.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p1.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p0.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p0.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p0.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p2.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p2.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p2.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p2.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p3.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p3.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p3.y, p2.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p3.y, p2.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p1.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p1.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p1.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p1.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p0.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p0.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p0.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p0.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p2.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p2.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p2.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p2.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    self.val_coord_3d_fast_vec(offset, ivec3(p1.x, p3.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p0.x, p3.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p2.x, p3.y, p3.z)),
                    self.val_coord_3d_fast_vec(offset, ivec3(p3.x, p3.y, p3.z)),
                    p5.x,
                ),
                p5.y,
            ),
            p5.z,
        ) * CUBIC_3D_BOUNDING
    }

    #[allow(dead_code)]
    fn get_cubic_fractal(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.fractal_type {
            FractalType::FBM => self.single_cubic_fractal_fbm(x, y),
            FractalType::Billow => self.single_cubic_fractal_billow(x, y),
            FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi(x, y),
        }
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
            sum -= (1.0 - fast_abs_f(self.single_cubic(self.perm[i as usize], x, y))) * amp;
            i += 1;
        }

        sum
    }

    #[allow(dead_code)]
    fn get_cubic(&self, x: f32, y: f32) -> f32 {
        self.single_cubic(0, x * self.frequency, y * self.frequency)
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

    #[allow(dead_code)]
    // Cellular Noise
    fn get_cellular3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;
        z *= self.frequency;

        match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular3d(x, y, z),
            CellularReturnType::Distance => self.single_cellular3d(x, y, z),
            _ => self.single_cellular_2edge3d(x, y, z),
        }
    }

    fn single_cellular3d(&self, x: f32, y: f32, z: f32) -> f32 {
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
                            let lut_pos: u8 = self.index3d_256(0, xi, yi, zi);

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
                            let lut_pos: u8 = self.index3d_256(0, xi, yi, zi);

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
                            let lut_pos: u8 = self.index3d_256(0, xi, yi, zi);

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
            CellularReturnType::CellValue => self.val_coord_3d(self.seed as i32, xc, yc, zc),
            CellularReturnType::Distance => distance,
            _ => 0.0,
        }
    }

    fn single_cellular_2edge3d(&self, x: f32, y: f32, z: f32) -> f32 {
        let xr = fast_round(x);
        let yr = fast_round(y);
        let zr = fast_round(z);

        let mut distance: Vec<f32> = vec![999999.0; FN_CELLULAR_INDEX_MAX + 1];
        //FN_DECIMAL distance[FN_CELLULAR_INDEX_MAX+1] = { 999999,999999,999999,999999 };

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let lut_pos: u8 = self.index3d_256(0, xi, yi, zi);

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
                            let lut_pos = self.index3d_256(0, xi, yi, zi);

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
                            let lut_pos = self.index3d_256(0, xi, yi, zi);

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
    fn get_cellular(&self, mut x: f32, mut y: f32) -> f32 {
        x *= self.frequency;
        y *= self.frequency;

        match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular(x, y),
            CellularReturnType::Distance => self.single_cellular(x, y),
            _ => self.single_cellular_2edge(x, y),
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

    #[allow(dead_code)]
    fn gradient_perturb3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        self.single_gradient_perturb3d(0, self.gradient_perturb_amp, self.frequency, x, y, z);
    }

    #[allow(dead_code)]
    fn gradient_perturb_fractal3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        let mut amp = self.gradient_perturb_amp * self.fractal_bounding;
        let mut freq = self.frequency;
        let mut i = 1;

        self.single_gradient_perturb3d(self.perm[0], amp, self.frequency, x, y, z);

        while i < self.octaves {
            freq *= self.lacunarity;
            amp *= self.gain;
            self.single_gradient_perturb3d(self.perm[i as usize], amp, freq, x, y, z);

            i += 1;
        }
    }

    #[allow(dead_code)]
    fn single_gradient_perturb3d(
        &self,
        offset: u8,
        warp_amp: f32,
        frequency: f32,
        x: &mut f32,
        y: &mut f32,
        z: &mut f32,
    ) {
        let xf = *x * frequency;
        let yf = *y * frequency;
        let zf = *z * frequency;

        let x0 = fast_floor(xf);
        let y0 = fast_floor(yf);
        let z0 = fast_floor(zf);
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let z1 = z0 + 1;

        let xs: f32;
        let ys: f32;
        let zs: f32;
        match self.interp {
            Interp::Linear => {
                xs = xf - x0 as f32;
                ys = yf - y0 as f32;
                zs = zf - z0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(xf - x0 as f32);
                ys = interp_hermite_func(yf - y0 as f32);
                zs = interp_hermite_func(zf - z0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(xf - x0 as f32);
                ys = interp_quintic_func(yf - y0 as f32);
                zs = interp_quintic_func(zf - z0 as f32);
            }
        }

        let mut lut_pos0 = self.index3d_256(offset, x0, y0, z0);
        let mut lut_pos1 = self.index3d_256(offset, x1, y0, z0);

        let mut lx0x = lerp(
            CELL_3D_X[lut_pos0 as usize],
            CELL_3D_X[lut_pos1 as usize],
            xs,
        );
        let mut ly0x = lerp(
            CELL_3D_Y[lut_pos0 as usize],
            CELL_3D_Y[lut_pos1 as usize],
            xs,
        );
        let mut lz0x = lerp(
            CELL_3D_Z[lut_pos0 as usize],
            CELL_3D_Z[lut_pos1 as usize],
            xs,
        );

        lut_pos0 = self.index3d_256(offset, x0, y1, z0);
        lut_pos1 = self.index3d_256(offset, x1, y1, z0);

        let mut lx1x = lerp(
            CELL_3D_X[lut_pos0 as usize],
            CELL_3D_X[lut_pos1 as usize],
            xs,
        );
        let mut ly1x = lerp(
            CELL_3D_Y[lut_pos0 as usize],
            CELL_3D_Y[lut_pos1 as usize],
            xs,
        );
        let mut lz1x = lerp(
            CELL_3D_Z[lut_pos0 as usize],
            CELL_3D_Z[lut_pos1 as usize],
            xs,
        );

        let lx0y = lerp(lx0x, lx1x, ys);
        let ly0y = lerp(ly0x, ly1x, ys);
        let lz0y = lerp(lz0x, lz1x, ys);

        lut_pos0 = self.index3d_256(offset, x0, y0, z1);
        lut_pos1 = self.index3d_256(offset, x1, y0, z1);

        lx0x = lerp(
            CELL_3D_X[lut_pos0 as usize],
            CELL_3D_X[lut_pos1 as usize],
            xs,
        );
        ly0x = lerp(
            CELL_3D_Y[lut_pos0 as usize],
            CELL_3D_Y[lut_pos1 as usize],
            xs,
        );
        lz0x = lerp(
            CELL_3D_Z[lut_pos0 as usize],
            CELL_3D_Z[lut_pos1 as usize],
            xs,
        );

        lut_pos0 = self.index3d_256(offset, x0, y1, z1);
        lut_pos1 = self.index3d_256(offset, x1, y1, z1);

        lx1x = lerp(
            CELL_3D_X[lut_pos0 as usize],
            CELL_3D_X[lut_pos1 as usize],
            xs,
        );
        ly1x = lerp(
            CELL_3D_Y[lut_pos0 as usize],
            CELL_3D_Y[lut_pos1 as usize],
            xs,
        );
        lz1x = lerp(
            CELL_3D_Z[lut_pos0 as usize],
            CELL_3D_Z[lut_pos1 as usize],
            xs,
        );

        *x += lerp(lx0y, lerp(lx0x, lx1x, ys), zs) * warp_amp;
        *y += lerp(ly0y, lerp(ly0x, ly1x, ys), zs) * warp_amp;
        *z += lerp(lz0y, lerp(lz0x, lz1x, ys), zs) * warp_amp;
    }

    #[allow(dead_code)]
    fn gradient_perturb(&self, x: &mut f32, y: &mut f32) {
        self.single_gradient_perturb(0, self.gradient_perturb_amp, self.frequency, x, y);
    }

    #[allow(dead_code)]
    fn gradient_perturb_fractal(&self, x: &mut f32, y: &mut f32) {
        let mut amp = self.gradient_perturb_amp * self.fractal_bounding;
        let mut freq = self.frequency;
        let mut i = 1;

        self.single_gradient_perturb(self.perm[0], amp, self.frequency, x, y);

        while i < self.octaves {
            freq *= self.lacunarity;
            amp *= self.gain;
            self.single_gradient_perturb(self.perm[i as usize], amp, freq, x, y);
            i += 1;
        }
    }

    #[allow(dead_code)]
    fn single_gradient_perturb(
        &self,
        offset: u8,
        warp_amp: f32,
        frequency: f32,
        x: &mut f32,
        y: &mut f32,
    ) {
        let xf = *x * frequency;
        let yf = *y * frequency;

        let x0 = fast_floor(xf);
        let y0 = fast_floor(yf);
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let xs: f32;
        let ys: f32;
        match self.interp {
            Interp::Linear => {
                xs = xf - x0 as f32;
                ys = yf - y0 as f32;
            }
            Interp::Hermite => {
                xs = interp_hermite_func(xf - x0 as f32);
                ys = interp_hermite_func(yf - y0 as f32);
            }
            Interp::Quintic => {
                xs = interp_quintic_func(xf - x0 as f32);
                ys = interp_quintic_func(yf - y0 as f32);
            }
        }

        let mut lut_pos0 = self.index2d_256(offset, x0, y0);
        let mut lut_pos1 = self.index2d_256(offset, x1, y0);

        let lx0x = lerp(
            CELL_2D_X[lut_pos0 as usize],
            CELL_2D_X[lut_pos1 as usize],
            xs,
        );
        let ly0x = lerp(
            CELL_2D_Y[lut_pos0 as usize],
            CELL_2D_Y[lut_pos1 as usize],
            xs,
        );

        lut_pos0 = self.index2d_256(offset, x0, y1);
        lut_pos1 = self.index2d_256(offset, x1, y1);

        let lx1x = lerp(
            CELL_2D_X[lut_pos0 as usize],
            CELL_2D_X[lut_pos1 as usize],
            xs,
        );
        let ly1x = lerp(
            CELL_2D_Y[lut_pos0 as usize],
            CELL_2D_Y[lut_pos1 as usize],
            xs,
        );

        *x += lerp(lx0x, lx1x, ys) * warp_amp;
        *y += lerp(ly0x, ly1x, ys) * warp_amp;
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
