use glam::{Vec3A, ivec3, vec4};

use crate::{
    Builder, FractalType, Interp, fastnoise::{
        Sampler,
        utils::{fractal_bounding, interp_hermite_func_vec, interp_quintic_func_vec, lerp, permutate, val_coord_3d_fast},
    }
};

#[derive(Clone, Copy, Default)]
pub struct ValueNoiseBuilder {
    pub fractal_type: Option<FractalType>,
    pub frequency: f32,
    pub gain: f32,
    pub interp: Interp,
    pub lacunarity: f32,
    pub octaves: u16,
    pub seed: u64,
}

impl Builder for ValueNoiseBuilder {
    type Output = ValueNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            fractal_bounding: fractal_bounding(self.gain, self.octaves),
            fractal_type: self.fractal_type,
            frequency: self.frequency,
            gain: self.gain,
            interp: self.interp,
            lacunarity: self.lacunarity,
            octaves: self.octaves as usize,
            perm: permutate(self.seed)[0],
        }
    }
}

#[derive(Clone, Copy)]
pub struct ValueNoise {
    fractal_bounding: f32,
    fractal_type: Option<FractalType>,
    frequency: f32,
    gain: f32,
    interp: Interp,
    lacunarity: f32,
    octaves: usize,
    perm: [u8; 512],
}

impl Sampler for ValueNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.fractal_type {
            Some(fractal) => match fractal {
                FractalType::FBM => self.single_value_fractal_fbm3d(pos),
                FractalType::Billow => self.single_value_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_value_fractal_rigid_multi3d(pos),
            },
            None => self.single_value3d(0, pos),
        }
    }
}

impl ValueNoise {
    fn single_value_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_value3d(self.perm[0], pos);
        let mut amp = 1.0;

        for i in 1..self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_value3d(self.perm[i], pos) * amp;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum = self.single_value3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp = 1.0;

        for i in 1..self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_value3d(self.perm[i], pos).abs().mul_add(2.0, -1.0) * amp;
        }

        sum * self.fractal_bounding
    }

    fn single_value_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum= 1.0 - self.single_value3d(self.perm[0], pos).abs();
        let mut amp = 1.0;

        for i in 1..self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum -= (1.0 - self.single_value3d(self.perm[i], pos).abs()) * amp;
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
            val_coord_3d_fast(&self.perm, offset, p0),
            val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p0.z)),
            val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p0.y, p1.z)),
            val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p1.z)),
        );
        let q1 = vec4(
            val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p0.z)),
            val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p1.y, p0.z)),
            val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p1.z)),
            val_coord_3d_fast(&self.perm, offset, p1),
        );

        let qf = q0.lerp(q1, ps.x);
        let yf0 = lerp(qf.x, qf.y, ps.y);
        let yf1 = lerp(qf.z, qf.w, ps.y);

        lerp(yf0, yf1, ps.z)
    }
}