use crate::activation::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        let sigmoids = self.function(x);
        sigmoids.iter().map(|x| x * (1.0 - x)).collect()
    }
}
