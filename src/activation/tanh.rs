use crate::activation::Activation;

pub struct Tanh;

impl Activation for Tanh {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|x| x.tanh()).collect()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        let tanh_values = self.function(x);
        tanh_values.iter().map(|x| 1.0 - x * x).collect()
    }
}
