use std::f32;

pub enum MixType {
    Average,
    Maximum,
    Minimum,
    // Random,
}

impl MixType {
    pub fn mix(&self, values: &[f32]) -> f32 {
        let iter = values.iter();
        match self {
            MixType::Average => iter.sum::<f32>() / values.len() as f32,
            MixType::Maximum => iter.fold(0.0, |acc, val| acc.max(*val)),
            MixType::Minimum => iter.fold(1.0, |acc, val| acc.min(*val)),
        }
    }
}

pub struct StrongF32 {
    inner: f32,
    strength: f32,
}

fn normalize(values: &[StrongF32]) -> Vec<f32> {

    let mut max_strength = 0f32;
    for value in values {
        max_strength = max_strength.max(value.strength);
    }
    let max_recip = max_strength.recip();

    let mut result = Vec::with_capacity(values.len());
    for value in values {
        result.push(value.inner * value.strength * max_recip);
    }

    result

}
