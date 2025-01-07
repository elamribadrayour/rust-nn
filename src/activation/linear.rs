use crate::activation::Activation;

pub struct Linear;

impl Activation for Linear {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.to_vec()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        vec![1.0; x.len()]
    }
}
