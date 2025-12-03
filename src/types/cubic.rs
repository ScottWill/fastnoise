use glam::{Vec3A, ivec3};

use crate::{Builder, Sampler, consts::CUBIC_3D_BOUNDING, utils::*};
use super::fractal::{FractalNoise, FractalNoiseBuilder};

#[derive(Clone, Copy, Debug, Default)]
pub struct CubicNoiseBuilder {
    pub fractal_noise: Option<FractalNoiseBuilder>,
    pub frequency: f32,
    pub seed: u64,
}

impl Builder for CubicNoiseBuilder {
    type Output = CubicNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            fractal_noise: self.fractal_noise.and_then(|v| Some(v.build())),
            frequency: self.frequency,
            perm: permutate(self.seed)[0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CubicNoise {
    fractal_noise: Option<FractalNoise>,
    frequency: f32,
    perm: [u8; 512],
}

impl From<CubicNoiseBuilder> for CubicNoise {
    fn from(value: CubicNoiseBuilder) -> Self {
        value.build()
    }
}

impl Sampler for CubicNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.fractal_noise {
            Some(fractal) => fractal.sample3d(pos, |offset, pos| {
                self.cubic3d(offset, pos)
            }),
            None => self.cubic3d(None, pos),
        }
    }
}

impl CubicNoise {
    fn cubic3d(&self, offset: Option<usize>, pos: Vec3A) -> f32 {
        let offset = match offset {
            Some(ix) => self.perm[ix],
            None => 0,
        };

        let p0 = pos.floor().as_ivec3();
        let p1 = p0 - 1;
        let p2 = p0 + 1;
        let p3 = p0 + 2;
        let p5 = pos - p0.as_vec3a();

        cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p1.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p1.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p1.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p0.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p0.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p0.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p2.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p2.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p2.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p2.y, p1.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p3.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p3.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p3.y, p1.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p3.y, p1.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p1.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p1.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p1.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p0.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p0.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p0.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p2.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p2.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p2.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p2.y, p0.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p3.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p3.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p3.y, p0.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p3.y, p0.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p1.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p1.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p1.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p0.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p0.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p0.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p2.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p2.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p2.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p2.y, p2.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p3.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p3.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p3.y, p2.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p3.y, p2.z)),
                    p5.x,
                ),
                p5.y,
            ),
            cubic_lerp(
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p1.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p1.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p1.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p1.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p0.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p0.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p0.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p0.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p2.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p2.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p2.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p2.y, p3.z)),
                    p5.x,
                ),
                cubic_lerp(
                    val_coord_3d_fast(&self.perm, offset, ivec3(p1.x, p3.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p0.x, p3.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p2.x, p3.y, p3.z)),
                    val_coord_3d_fast(&self.perm, offset, ivec3(p3.x, p3.y, p3.z)),
                    p5.x,
                ),
                p5.y,
            ),
            p5.z,
        ) * CUBIC_3D_BOUNDING
    }
}