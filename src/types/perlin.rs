use glam::{Vec2, Vec3A, Vec4Swizzles as _, ivec2, ivec3, vec2, vec3a, vec4};
#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "persistence")]
use serde_with::{serde_as, Bytes};

use crate::{Builder, Interp, Sampler, utils::*};
use super::fractal::{FractalNoise, FractalNoiseBuilder};

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct PerlinNoiseBuilder {
    pub amplitude: f32,
    pub fractal_noise: Option<FractalNoiseBuilder>,
    pub frequency: f32,
    pub interp: Interp,
    pub seed: u64,
}

impl Default for PerlinNoiseBuilder {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            fractal_noise: Default::default(),
            frequency: 1.0,
            interp: Default::default(),
            seed: Default::default(),
        }
    }
}

impl Builder for PerlinNoiseBuilder {
    type Output = PerlinNoise;
    fn build(self) -> Self::Output {
        let [perm, perm12] = permutate(self.seed);
        Self::Output {
            amplitude: self.amplitude,
            fractal_noise: self.fractal_noise
                .and_then(|v| Some(v.build())),
            frequency: self.frequency,
            interp: self.interp,
            perm,
            perm12,
        }
    }
}

#[cfg_attr(feature = "persistence", serde_as)]
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct PerlinNoise {
    amplitude: f32,
    fractal_noise: Option<FractalNoise>,
    frequency: f32,
    interp: Interp,
    #[cfg_attr(feature = "persistence", serde_as(as = "Bytes"))]
    perm: [u8; 512],
    #[cfg_attr(feature = "persistence", serde_as(as = "Bytes"))]
    perm12: [u8; 512],
}

impl From<PerlinNoiseBuilder> for PerlinNoise {
    fn from(value: PerlinNoiseBuilder) -> Self {
        value.build()
    }
}

impl Sampler for PerlinNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        let value = match self.fractal_noise {
            Some(fractal) => fractal.sample3d(pos, |offset, pos| {
                self.perlin3d(offset, pos)
            }),
            None => self.perlin3d(None, pos),
        };
        value * self.amplitude
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        let pos = position.into() * self.frequency;
        let value = match self.fractal_noise {
            Some(fractal) => fractal.sample2d(pos, |offset, pos| {
                self.perlin(offset, pos)
            }),
            None => self.perlin(None, pos),
        };
        value * self.amplitude
    }
}

impl PerlinNoise {
    fn perlin3d(&self, offset: Option<usize>, pos: Vec3A) -> f32 {
        let offset = match offset {
            Some(ix) => self.perm[ix],
            None => 0,
        };

        let p0 = pos.floor();
        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec3(pos - p0),
            Interp::Quintic => interp_quintic_func_vec3(pos - p0),
        };

        let d0 = pos - p0;
        let d1 = d0 - 1.0;

        let p0 = p0.as_ivec3();
        let p1 = p0 + 1;

        let q0 = vec4(
            grad_coord_3d(&self.perm, &self.perm12, offset, p0, d0),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p0.x, p1.y, p0.z), vec3a(d0.x, d1.y, d0.z)),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p0.x, p0.y, p1.z), vec3a(d0.x, d0.y, d1.z)),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p0.x, p1.y, p1.z), vec3a(d0.x, d1.y, d1.z)),
        );
        let q1 = vec4(
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p1.x, p0.y, p0.z), vec3a(d1.x, d0.y, d0.z)),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p1.x, p1.y, p0.z), vec3a(d1.x, d1.y, d0.z)),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p1.x, p0.y, p1.z), vec3a(d1.x, d0.y, d1.z)),
            grad_coord_3d(&self.perm, &self.perm12, offset, ivec3(p1.x, p1.y, p1.z), d1),
        );

        let qf0 = q0.lerp(q1, ps.x);
        let qf1 = qf0.xz().lerp(qf0.yw(), ps.y);

        lerp(qf1.x, qf1.y, ps.z)

    }

    fn perlin(&self, offset: Option<usize>, pos: Vec2) -> f32 {
        let offset = match offset {
            Some(ix) => self.perm[ix],
            None => 0,
        };

        let p0 = pos.floor();

        let ps = match self.interp {
            Interp::Linear => pos - p0,
            Interp::Hermite => interp_hermite_func_vec2(pos - p0),
            Interp::Quintic => interp_quintic_func_vec2(pos - p0),
        };

        let d0 = pos - p0;
        let d1 = d0 - 1.0;

        let p0 = p0.as_ivec2();
        let p1 = p0 + 1;

        let xf0 = vec2(
            grad_coord_2d(&self.perm, &self.perm12, offset, p0, d0),
            grad_coord_2d(&self.perm, &self.perm12, offset, ivec2(p0.x, p1.y), vec2(d0.x, d1.y)),
        );
        let xf1 = vec2(
            grad_coord_2d(&self.perm, &self.perm12, offset, ivec2(p1.x, p0.y), vec2(d1.x, d0.y)),
            grad_coord_2d(&self.perm, &self.perm12, offset, p1, d1),
        );
        let xff = xf0.lerp(xf1, ps.x);

        lerp(xff.x, xff.y, ps.y)

    }
}