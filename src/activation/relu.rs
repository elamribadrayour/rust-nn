use crate::activation::Activation;

pub struct ReLU;

impl Activation for ReLU {
    fn function(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
