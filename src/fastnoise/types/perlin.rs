use glam::{Vec3A, ivec3, vec3a, vec4};

use crate::{
    Builder, FractalType, Interp, fastnoise::{
        Sampler,
        utils::{fractal_bounding, grad_coord_3d, interp_hermite_func_vec, interp_quintic_func_vec, lerp, permutate},
    }
};

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

impl Sampler for PerlinNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.fractal_type {
            Some(fractal) => match fractal {
                FractalType::FBM => self.single_perlin_fractal_fbm3d(pos),
                FractalType::Billow => self.single_perlin_fractal_billow3d(pos),
                FractalType::RigidMulti => self.single_perlin_fractal_rigid_multi3d(pos),
            },
            None => self.single_perlin3d(0, pos),
        }
    }
}

impl PerlinNoise {
    fn single_perlin_fractal_fbm3d(&self, mut pos: Vec3A) -> f32 {
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

    fn single_perlin_fractal_billow3d(&self, mut pos: Vec3A) -> f32 {
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

    fn single_perlin_fractal_rigid_multi3d(&self, mut pos: Vec3A) -> f32 {
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
            Interp::Hermite => interp_hermite_func_vec(pos - p0),
            Interp::Quintic => interp_quintic_func_vec(pos - p0),
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

        let qf = q0.lerp(q1, ps.x);
        let yf0 = lerp(qf.x, qf.y, ps.y);
        let yf1 = lerp(qf.z, qf.w, ps.y);

        lerp(yf0, yf1, ps.z)
    }
}