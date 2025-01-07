use std::f64;

use crate::loss::Loss;

pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn function(&self, outputs: &[f64], targets: &[f64]) -> f64 {
        -targets
            .iter()
            .zip(outputs.iter())
            .map(|(t, p)| {
                let p = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
                t * p.ln()
            })
            .sum::<f64>()
    }

    fn gradient(&self, outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        targets
            .iter()
            .zip(outputs.iter())
            .map(|(t, p)| (p - t))
            .collect()
    }
}
