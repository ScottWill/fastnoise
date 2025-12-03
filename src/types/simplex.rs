use glam::{Vec3A, vec4};

use crate::{Builder, Sampler, consts::*, utils::*};
use super::fractal::{FractalNoise, FractalNoiseBuilder};

#[derive(Clone, Copy, Debug, Default)]
pub struct SimplexNoiseBuilder {
    pub fractal_noise: Option<FractalNoiseBuilder>,
    pub frequency: f32,
    pub seed: u64,
}

impl Builder for SimplexNoiseBuilder {
    type Output = SimplexNoise;
    fn build(self) -> Self::Output {
        let [perm, perm12] = permutate(self.seed);
        Self::Output {
            fractal_noise: self.fractal_noise.and_then(|v| Some(v.build())),
            frequency: self.frequency,
            perm,
            perm12,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimplexNoise {
    fractal_noise: Option<FractalNoise>,
    frequency: f32,
    perm: [u8; 512],
    perm12: [u8; 512],
}

impl From<SimplexNoiseBuilder> for SimplexNoise {
    fn from(value: SimplexNoiseBuilder) -> Self {
        value.build()
    }
}

impl Sampler for SimplexNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.fractal_noise {
            Some(fractal) => fractal.sample3d(pos, |offset, pos| {
                self.simplex3d(offset, pos)
            }),
            None => self.simplex3d(None, pos),
        }
    }
}

impl SimplexNoise {
    fn simplex3d(&self, offset: Option<usize>, p: Vec3A) -> f32 {
        let offset = match offset {
            Some(ix) => self.perm[ix],
            None => 0,
        };

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
                t * t * grad_coord_3d(&self.perm, &self.perm12, offset, q, p0)
            }
        };

        t = 0.6 - (p1 * p1).element_sum();
        let n1 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * grad_coord_3d(&self.perm, &self.perm12, offset, q + q1, p1)
            }
        };

        t = 0.6 - (p2 * p2).element_sum();
        let n2 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * grad_coord_3d(&self.perm, &self.perm12, offset, q + q2, p2)
            }
        };

        t = 0.6 - (p3 * p3).element_sum();
        let n3 = match t < 0.0 {
            true => 0.0,
            false => {
                t *= t;
                t * t * grad_coord_3d(&self.perm, &self.perm12, offset, q + 1, p3)
            }
        };

        32.0 * vec4(n0, n1, n2, n3).element_sum()
    }
}