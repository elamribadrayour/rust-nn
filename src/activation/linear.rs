use crate::activation::Activation;

pub struct Linear;

impl Activation for Linear {
    fn function(&self, x: f64) -> f64 {
        x
    }

    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}
