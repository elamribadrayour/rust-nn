pub trait Loss {
    fn function(&self, outputs: &[f64], targets: &[f64]) -> f64;
    fn gradient(&self, outputs: &[f64], targets: &[f64]) -> Vec<f64>;
}
