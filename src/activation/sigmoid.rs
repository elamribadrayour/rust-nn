use crate::activation::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn function(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }
}
