use crate::loss::Loss;

pub struct Mse;

impl Loss for Mse {
    fn function(&self, outputs: &[f64], targets: &[f64]) -> f64 {
        let n = outputs.len() as f64;
        let sum_of_squares = outputs
            .iter()
            .zip(targets.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>();

        sum_of_squares / n
    }

    fn gradient(&self, outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        let n = outputs.len() as f64;
        outputs
            .iter()
            .zip(targets.iter())
            .map(|(o, t)| 2.0 * (o - t) / n)
            .collect()
    }
}
