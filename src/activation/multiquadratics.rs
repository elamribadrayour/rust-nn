use std::collections::HashMap;

use crate::activation::Activation;

pub struct Multiquadratics {
    pub mu: f64,
    pub a: f64,
}

impl Multiquadratics {
    pub fn new(params: HashMap<String, f64>) -> Self {
        Self {
            a: *params.get("a").unwrap_or(&1.0),
            mu: *params.get("mu").unwrap_or(&0.0),
        }
    }
}

impl Activation for Multiquadratics {
    fn function(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|x| ((x - self.mu).powi(2) + self.a.powi(2)).sqrt())
            .collect()
    }

    fn derivative(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|x| (x - self.mu) / ((x - self.mu).powi(2) + self.a.powi(2)).sqrt())
            .collect()
    }
}
