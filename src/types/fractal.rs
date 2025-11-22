use glam::{Vec2, Vec3A};

use crate::{Builder, FractalType, utils::fractal_bounding};

#[derive(Clone, Copy, Default)]
pub struct FractalNoiseBuilder {
    pub fractal_type: FractalType,
    pub gain: f32,
    pub lacunarity: f32,
    pub octaves: u16,
}

impl Builder for FractalNoiseBuilder {
    type Output = FractalNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            fractal_bounding: fractal_bounding(self.gain, self.octaves),
            fractal_type: self.fractal_type,
            gain: self.gain,
            lacunarity: self.lacunarity,
            octaves: self.octaves as _,
        }
    }
}

#[derive(Clone, Copy)]
pub struct FractalNoise {
    pub fractal_bounding: f32,
    pub fractal_type: FractalType,
    pub gain: f32,
    pub lacunarity: f32,
    pub octaves: usize,
}

impl FractalNoise {
    pub(super) fn sample3d<F>(&self, pos: Vec3A, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec3A) -> f32
    {
        match self.fractal_type {
            FractalType::Billow => self.billow3d(pos, noise_fn),
            FractalType::FBM => self.fbm3d(pos, noise_fn),
            FractalType::RigidMulti => self.rigid3d(pos, noise_fn),
            FractalType::None => noise_fn(None, pos),
        }
    }

    fn billow3d<F>(&self, mut pos: Vec3A, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec3A) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn fbm3d<F>(&self, mut pos: Vec3A, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec3A) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn rigid3d<F>(&self, mut pos: Vec3A, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec3A) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos).abs() * amp;
            i += 1;
        }

        sum
    }

    //2d
    pub(super) fn sample2d<F>(&self, pos: Vec2, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec2) -> f32
    {
        match self.fractal_type {
            FractalType::Billow => self.billow2d(pos, noise_fn),
            FractalType::FBM => self.fbm2d(pos, noise_fn),
            FractalType::RigidMulti => self.rigid2d(pos, noise_fn),
            FractalType::None => noise_fn(None, pos),
        }
    }

    fn billow2d<F>(&self, mut pos: Vec2, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec2) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn fbm2d<F>(&self, mut pos: Vec2, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec2) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn rigid2d<F>(&self, mut pos: Vec2, noise_fn: F) -> f32
    where F: Fn(Option<usize>, Vec2) -> f32
    {
        let mut sum: f32 = noise_fn(Some(0), pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += noise_fn(Some(i), pos).abs() * amp;
            i += 1;
        }

        sum
    }
}