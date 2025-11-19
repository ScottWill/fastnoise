use glam::{Vec3A, ivec3, vec4};

use crate::{
    FractalType,
    Interp,
    fastnoise::{
        Sampler,
        utils::{interp_hermite_func_vec, interp_quintic_func_vec, lerp, val_coord_3d_fast},
    },
};

pub struct ValueNoise {
    fractal_bounding: f32,
    fractal_type: Option<FractalType>,
    frequency: f32,
    gain: f32,
    interp: Interp,
    lacunarity: f32,
    octaves: i32,
    perm: [u8; 256],
    seed: f32,
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