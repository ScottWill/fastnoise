#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Default, PartialEq, Copy, Clone)]
/// Type of noise to generate
pub enum NoiseType {
    Cellular,
    Cubic,
    Perlin,
    Simplex,
    #[default]
    Value,
    White,
}

impl Display for NoiseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Default, PartialEq, Copy, Clone)]
/// Interpolation function to use
pub enum Interp {
    #[default]
    Linear,
    Hermite,
    Quintic,
}

impl Display for Interp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Default, PartialEq, Copy, Clone)]
/// Fractal function to use
pub enum FractalType {
    #[default]
    None,
    FBM,
    Billow,
    RigidAlt,
    RigidMulti,
}

impl Display for FractalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Default, PartialEq, Copy, Clone)]
/// Cellular noise distance function to use
pub enum CellularDistanceFunction {
    #[default]
    Euclidean,
    Manhattan,
    Natural,
}

impl Display for CellularDistanceFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Default, PartialEq, Copy, Clone)]
/// What type of cellular noise result do you want
pub enum CellularReturnType {
    #[default]
    CellValue,
    Distance,
}

impl Display for CellularReturnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}