use glam::{Vec3A, ivec3};

use crate::{FractalType, fastnoise::{Sampler, consts::CUBIC_3D_BOUNDING, utils::{cubic_lerp, val_coord_3d_fast}}};

pub struct CubicNoise {
    fractal_bounding: f32,
    fractal_type: Option<FractalType>,
    frequency: f32,
    gain: f32,
    lacunarity: f32,
    octaves: i32,
    perm: [u8; 256],
    seed: f32,
}

impl Sampler for CubicNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.fractal_type {
            Some(fractal) => match fractal {
                FractalType::FBM => self.single_cubic_fractal_fbm3d(pos),
                FractalType::Billow => self.single_cubic_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_cubic_fractal_rigid_multi3d(pos),
            },
            None => self.single_cubic3d(0, pos),
        }
    }
}

impl CubicNoise {
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