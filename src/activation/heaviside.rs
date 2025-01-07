use crate::activation::Activation;

pub struct Heaviside;

impl Activation for Heaviside {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect()
    }
}
