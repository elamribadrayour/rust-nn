use std::collections::HashMap;

use crate::activation::Activation;

pub struct Gaussian {
    pub mu: f64,
    pub sigma: f64,
}

impl Gaussian {
    pub fn new(params: HashMap<String, f64>) -> Self {
        Self {
            mu: *params.get("mu").unwrap_or(&0.0),
            sigma: *params.get("sigma").unwrap_or(&1.0),
        }
    }
}

impl Activation for Gaussian {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|x| ((x - self.mu).powi(2) / (2.0 * self.sigma.powi(2))).exp())
            .collect()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|x| {
                -((x - self.mu) / self.sigma.powi(2))
                    * ((x - self.mu).powi(2) / (2.0 * self.sigma.powi(2))).exp()
            })
            .collect()
    }
}
