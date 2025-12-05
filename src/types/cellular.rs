use glam::{IVec3, Vec2, Vec3A, ivec2, ivec3};

#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "persistence")]
use serde_with::{serde_as, Bytes};

use crate::{Builder, CellularDistanceFunction, CellularReturnType, Sampler, consts::*, utils::*};

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct CellularNoiseBuilder {
    pub amplitude: f32,
    pub cellular_distance_function: CellularDistanceFunction,
    pub cellular_jitter: f32,
    pub cellular_return_type: CellularReturnType,
    pub frequency: f32,
    pub seed: u64,
}

impl Default for CellularNoiseBuilder {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            cellular_distance_function: Default::default(),
            cellular_jitter: Default::default(),
            cellular_return_type: Default::default(),
            frequency: 1.0,
            seed: Default::default(),
        }
    }
}

impl Builder for CellularNoiseBuilder {
    type Output = CellularNoise;
    fn build(self) -> Self::Output {
        Self::Output {
            amplitude: self.amplitude,
            cellular_distance_function: self.cellular_distance_function,
            cellular_jitter: self.cellular_jitter,
            cellular_return_type: self.cellular_return_type,
            frequency: self.frequency,
            perm: permutate(self.seed)[0],
            seed: self.seed as i32,
        }
    }
}

#[cfg_attr(feature = "persistence", serde_as)]
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellularNoise {
    amplitude: f32,
    cellular_distance_function: CellularDistanceFunction,
    cellular_jitter: f32,
    cellular_return_type: CellularReturnType,
    frequency: f32,
    #[cfg_attr(feature = "persistence", serde_as(as = "Bytes"))]
    perm: [u8; 512],
    seed: i32,
}

impl Sampler for CellularNoise {
    fn sample3d<V>(&self, position: V) -> f32 where V: Into<glam::Vec3A> {
        let pos = position.into() * self.frequency;
        let value = match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular3d(pos),
            CellularReturnType::Distance => self.single_cellular3d(pos),
        };
        value * self.amplitude
    }

    fn sample2d<P>(&self, position: P) -> f32 where P: Into<glam::Vec2> {
        let pos = position.into() * self.frequency;
        let value = match self.cellular_return_type {
            CellularReturnType::CellValue => self.single_cellular(pos),
            CellularReturnType::Distance => self.single_cellular(pos),
        };
        value * self.amplitude
    }

}

impl CellularNoise {

    fn single_cellular(&self, pos: Vec2) -> f32 {
        let [x, y] = pos.to_array();
        let [xr, yr] = pos.round().as_ivec2().to_array();

        let mut distance: f32 = 999999.0;

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = index2d_256(&self.perm, 0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = vec_x * vec_x + vec_y * vec_y;

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = index2d_256(&self.perm, 0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = fast_abs_f(vec_x) + fast_abs_f(vec_y);

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
            CellularDistanceFunction::Natural => {
                for xi in xr - 1..xr + 2 {
                    for yi in yr - 1..yr + 2 {
                        let lut_pos: u8 = index2d_256(&self.perm, 0, ivec2(xi, yi));

                        let vec_x =
                            xi as f32 - x + CELL_2D_X[lut_pos as usize] * self.cellular_jitter;
                        let vec_y =
                            yi as f32 - y + CELL_2D_Y[lut_pos as usize] * self.cellular_jitter;

                        let new_distance = (fast_abs_f(vec_x) + fast_abs_f(vec_y))
                            + (vec_x * vec_x + vec_y * vec_y);

                        if new_distance < distance {
                            distance = new_distance;
                        }
                    }
                }
            }
        }

        //let lut_pos : u8;
        match self.cellular_return_type {
            CellularReturnType::CellValue => {
                val_coord_2d(self.seed as i32, pos.as_ivec2())
            }
            _ => 0.0,
        }
    }

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
        }
    }
}