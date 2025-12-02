use glam::{Vec3A, ivec3, vec4};

use crate::{Builder, Interp, Sampler, utils::*};
use super::fractal::{FractalNoise, FractalNoiseBuilder};

#[derive(Clone, Copy, Debug, Default)]
pub struct ValueNoiseBuilder {
    pub fractal_noise: Option<FractalNoiseBuilder>,
    pub frequency: f32,
    pub interp: Interp,
    pub seed: u64,
}

impl Builder for ValueNoiseBuilder {
    type Output = ValueNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            fractal_noise: self.fractal_noise.and_then(|v| Some(v.build())),
            frequency: self.frequency,
            interp: self.interp,
            perm: permutate(self.seed)[0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ValueNoise {
    fractal_noise: Option<FractalNoise>,
    frequency: f32,
    interp: Interp,
    perm: [u8; 512],
}

impl From<ValueNoiseBuilder> for ValueNoise {
    fn from(value: ValueNoiseBuilder) -> Self {
        value.build()
    }
}

impl Sampler for ValueNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        let sample = match self.fractal_noise {
            Some(fractal) => fractal.sample3d(pos, |offset, pos| {
                self.value3d(offset, pos)
            }),
            None => self.value3d(None, pos),
        };
        normalize(sample, 1.0, 0.5)
    }
}

impl ValueNoise {
    fn value3d(&self, offset: Option<usize>, pos: Vec3A) -> f32 {
        let offset = match offset {
            Some(ix) => self.perm[ix],
            None => 0,
        };

        let p0 = pos.floor();
        let p1 = (p0 + 1.0).as_ivec3();
        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec3(pos - p0),
            Interp::Quintic => interp_quintic_func_vec3(pos - p0),
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