use glam::{IVec3, Vec3A, ivec3};

use crate::{
    Builder, CellularDistanceFunction, CellularReturnType, fastnoise::{
        Sampler,
        consts::{CELL_3D, FN_CELLULAR_INDEX_MAX},
        utils::{index3d_256, permutate, val_coord_3d},
    }
};

#[derive(Clone, Copy, Default)]
pub struct CellularNoiseBuilder {
    pub cellular_distance_function: CellularDistanceFunction,
    pub cellular_distance_index: (i32, i32),
    pub cellular_jitter: f32,
    pub cellular_return_type: CellularReturnType,
    pub frequency: f32,
    pub seed: u64,
}

impl Builder for CellularNoiseBuilder {
    type Output = CellularNoise;
    fn build(self) -> Self::Output {
        let cellular_distance_index = {
            let (a, b) = self.cellular_distance_index;
            (
                a.min(b).clamp(0, FN_CELLULAR_INDEX_MAX as i32),
                a.max(b).clamp(0, FN_CELLULAR_INDEX_MAX as i32),
            )
        };
        Self::Output {
            cellular_distance_function: self.cellular_distance_function,
            cellular_distance_index: cellular_distance_index,
            cellular_jitter: self.cellular_jitter,
            cellular_return_type: self.cellular_return_type,
            frequency: self.frequency,
            perm: permutate(self.seed)[0],
            seed: self.seed as i32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CellularNoise {
    cellular_distance_function: CellularDistanceFunction,
    cellular_distance_index: (i32, i32),
    cellular_jitter: f32,
    cellular_return_type: CellularReturnType,
    frequency: f32,
    perm: [u8; 512],
    seed: i32,
}

impl Sampler for CellularNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular3d(pos),
            CellularReturnType::Distance => self.single_cellular3d(pos),
            _ => self.single_cellular_2edge3d(pos),
        }
    }
}

impl CellularNoise {
    fn single_cellular3d(&self, pos: Vec3A) -> f32 {
        let [xr, yr, zr] = pos.as_ivec3().to_array();

        let mut distance: f32 = f32::MAX;
        let mut c = IVec3::ZERO;
        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = (vec * vec).element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum() + (vec * vec).element_sum();
                            if new_distance < distance {
                                distance = new_distance;
                                c = i;
                            }
                        }
                    }
                }
            }
        }

        match self.cellular_return_type {
            CellularReturnType::CellValue => val_coord_3d(self.seed, c),
            CellularReturnType::Distance => distance,
            _ => 0.0,
        }
    }

    fn single_cellular_2edge3d(&self, pos: Vec3A) -> f32 {
        let [xr, yr, zr] = pos.as_ivec3().to_array();

        let mut distance: Vec<f32> = vec![f32::MAX; FN_CELLULAR_INDEX_MAX + 1];

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = (vec * vec).element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        for zi in zr - 1..zr + 2 {
                            let i = ivec3(xi, yi, zi);
                            let lut_pos: u8 = index3d_256(&self.perm, 0, i);
                            let cell = CELL_3D[lut_pos as usize];
                            let vec = i.as_vec3a() - pos + cell * self.cellular_jitter;
                            let new_distance = vec.abs().element_sum() + (vec * vec).element_sum();
                            for i in (0..self.cellular_distance_index.1).rev() {
                                let min = distance[i as usize];
                                let max = distance[i as usize - 1];
                                distance[i as usize] = f32::clamp(new_distance, min, max);
                            }
                            distance[0] = f32::min(distance[0], new_distance);
                        }
                    }
                }
            }
        }

        match self.cellular_return_type {
            CellularReturnType::Distance2 => distance[self.cellular_distance_index.1 as usize],
            CellularReturnType::Distance2Add => {
                distance[self.cellular_distance_index.1 as usize]
                    + distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Sub => {
                distance[self.cellular_distance_index.1 as usize]
                    - distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Mul => {
                distance[self.cellular_distance_index.1 as usize]
                    * distance[self.cellular_distance_index.0 as usize]
            }
            CellularReturnType::Distance2Div => {
                distance[self.cellular_distance_index.0 as usize]
                    / distance[self.cellular_distance_index.1 as usize]
            }
            _ => 0.0,
        }
    }
}