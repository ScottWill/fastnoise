use glam::{Vec2, Vec3A, Vec4Swizzles as _, ivec2, ivec3, vec2, vec3a, vec4};

use crate::{Builder, FractalType, Interp, Sampler, utils::*};

#[derive(Clone, Copy, Default)]
pub struct PerlinNoiseBuilder {
    pub fractal_type: Option<FractalType>,
    pub frequency: f32,
    pub gain: f32,
    pub interp: Interp,
    pub lacunarity: f32,
    pub octaves: u16,
    pub seed: u64,
}

impl Builder for PerlinNoiseBuilder {
    type Output = PerlinNoise;
    fn build(self) -> Self::Output {
        let [perm, perm12] = permutate(self.seed);
        Self::Output {
            fractal_bounding: fractal_bounding(self.gain, self.octaves),
            fractal_type: self.fractal_type,
            frequency: self.frequency,
            gain: self.gain,
            interp: self.interp,
            lacunarity: self.lacunarity,
            octaves: self.octaves as usize,
            perm,
            perm12,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PerlinNoise {
    fractal_bounding: f32,
    fractal_type: Option<FractalType>,
    frequency: f32,
    gain: f32,
    interp: Interp,
    lacunarity: f32,
    octaves: usize,
    perm: [u8; 512],
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
        match self.fractal_type {
            Some(fractal) => match fractal {
                FractalType::FBM => self.fbm3d(pos),
                FractalType::Billow => self.billow3d(pos),
                FractalType::RigidMulti => self.rigid_multi3d(pos),
            },
            None => self.single_perlin3d(0, pos),
        }
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        let pos = position.into() * self.frequency;
        match self.fractal_type {
            Some(fractal) => match fractal {
                FractalType::FBM => self.fbm(pos),
                FractalType::Billow => self.billow(pos),
                FractalType::RigidMulti => self.rigid_multi(pos),
            },
            None => self.single_perlin(0, pos),
        }
    }
}

impl PerlinNoise {
    fn fbm3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_perlin3d(self.perm[0], pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d(self.perm[i], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn billow3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = self.single_perlin3d(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin3d(self.perm[i], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
        let mut sum: f32 = 1.0 - self.single_perlin3d(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_perlin3d(self.perm[i], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    fn single_perlin3d(&self, offset: u8, pos: Vec3A) -> f32 {
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

    fn fbm(&self, mut pos: Vec2) -> f32 {
        let mut sum: f32 = self.single_perlin(self.perm[0], pos);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin(self.perm[i], pos) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn billow(&self, mut pos: Vec2) -> f32 {
        let mut sum: f32 = self.single_perlin(self.perm[0], pos).abs().mul_add(2.0, -1.0);
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += self.single_perlin(self.perm[i], pos).abs().mul_add(2.0, -1.0) * amp;
            i += 1;
        }

        sum * self.fractal_bounding
    }

    fn rigid_multi(&self, mut pos: Vec2) -> f32 {
        let mut sum: f32 = 1.0 - self.single_perlin(self.perm[0], pos).abs();
        let mut amp: f32 = 1.0;
        let mut i = 1;

        while i < self.octaves {
            pos *= self.lacunarity;
            amp *= self.gain;
            sum += -(1.0 - self.single_perlin(self.perm[i], pos).abs()) * amp;
            i += 1;
        }

        sum
    }

    fn single_perlin(&self, offset: u8, pos: Vec2) -> f32 {
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